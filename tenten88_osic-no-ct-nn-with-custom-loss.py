# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.compose import ColumnTransformer 

from sklearn.model_selection import GroupKFold



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
BASE_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

BASE_PATIENT_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'

WORKING_DIR = '/kaggle/working/'

TEMP_DIR = '/kaggle/temp/'



NFOLDS = 5

TUNING = False   # when submission make it False
def compute_score(fvc_true, fvc_pred, confidence, return_vec=False):

    '''modified Laplace Log Likelihood'''

    sigma_clipped = np.maximum(confidence, 70.)

    delta = np.minimum(np.abs(fvc_true-fvc_pred), 1000.)

    metric = -np.sqrt(2)*delta/sigma_clipped - np.log(np.sqrt(2)*sigma_clipped)

    if return_vec:

        return metric

    return np.mean(metric)
Sex_mapper = {'Male':1, 'Female':0}





# prepare training data

def load_train():

    train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

    train_df['Percent'] /= 100.

    

    # delete duplicate rows by averaging FVC and Percent

    train_df[['FVC', 'Percent']] = train_df.groupby(['Patient', 'Weeks'])[['FVC', 'Percent']].transform('mean').values

    train_df.drop_duplicates(subset=['Patient', 'Weeks'], inplace=True)



    # set baseline weeks

    train_df['base_Weeks'] = train_df.groupby('Patient')['Weeks'].transform('min')

    train_df['Weeks_passed'] = train_df['Weeks'] - train_df['base_Weeks']

    

    # set baseline FVC and Percent

    base_df = train_df.loc[train_df.Weeks_passed==0, ['Patient', 'FVC', 'Percent']]

    base_df.columns = ['Patient', 'base_FVC', 'base_Percent']

    base_df.reset_index(drop=True, inplace=True)

    train_df = train_df.merge(base_df, on='Patient')



    train_df['ref_FVC'] = train_df['base_FVC'] / train_df['base_Percent'] 

    train_df['Sex'] = train_df['Sex'].map(Sex_mapper)

    train_df['target_ratio'] = train_df['FVC'] / train_df['base_FVC']

    

    # mark last 3 visits

    def f(x):

        result = np.zeros_like(x, dtype='int')

        result[-3:] = 1

        return result.astype(bool)



    train_df['last_3_visits'] = train_df.groupby('Patient')['FVC'].transform(f).values

    

    # add sample weights :

    train_df['weights'] =  1./train_df.groupby('Patient')['FVC'].transform('std')

    #train_df.loc[train_df.last_3_visits, 'weights'] = ... 

    

    train_df = train_df.reset_index(drop=True)

    

    return train_df 





# prepare test data

def load_test():

    test_df = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

    submit_df = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))



    test_df = test_df.rename(columns={'Weeks':'base_Weeks', 'FVC':'base_FVC', 'Percent':'base_Percent'})



    submit_df['Patient'] = submit_df.Patient_Week.str.split('_').str[0]

    submit_df['Weeks'] = submit_df.Patient_Week.str.split('_').str[1].astype(int)



    test_df = test_df.merge(submit_df, on='Patient')

    test_df['Weeks_passed'] = test_df['Weeks'] - test_df['base_Weeks']

    test_df['base_Percent'] /= 100.

    test_df['ref_FVC'] = test_df['base_FVC'] / test_df['base_Percent']

    test_df['Sex'] = test_df['Sex'].map(Sex_mapper)

    test_df = test_df.set_index('Patient_Week')

    return test_df, submit_df[['Patient_Week', 'FVC', 'Confidence']]
train_df = load_train()

test_df, submit_df = load_test()
submit_df.head()
test_df.head()
train_df.head()
import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras import layers, models

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import callbacks
def laplace_log_likelihood(y_true, y_pred):

    # y_pred=[fvc_ratio, log_sigma], y_true=[fvc_ratio, FVC_baseline]

    fvc_true = y_true[:, 0] * y_true[:,1]

    fvc_pred = y_pred[:, 0] * y_true[:,1]



    log_sigma = y_pred[:, 1]

    sigma = K.exp(log_sigma)



    sigma_clipped = K.maximum(sigma, K.constant(70., dtype='float32'))

    delta = K.minimum(K.abs(fvc_true - fvc_pred), K.constant(1000., dtype='float32'))



    sqrt2 = K.sqrt(K.constant(2, dtype='float32'))

    metric = -sqrt2*(delta/sigma_clipped) - K.log(sqrt2*sigma_clipped)



    return K.mean(metric)

    

def nll_gaussian_loss(y_true, y_pred):

    # negative loglilelihood loss (NLL) of gaussian

    # reference: https://www.kaggle.com/ttahara/osic-baseline-lgbm-with-custom-metric?scriptVersionId=38578211

    fvc_true = y_true[:, 0] * y_true[:,1]

    fvc_pred = y_pred[:, 0] * y_true[:, 1]

    log_sigma = y_pred[:, 1]



    term1 = -K.constant(0.5, dtype='float32') * K.square((fvc_true-fvc_pred)/K.exp(log_sigma))

    term2 = -K.log(K.sqrt(K.constant(2.0*np.pi, dtype='float32'))) - log_sigma

    loss = -(term1 + term2)

    

    # try sample weights

    if y_true.shape[1] == 3:

        weights = y_true[:, 2]

        return K.sum(loss * weights) / K.sum(weights)

    else:

        return K.mean(loss)





def create_model_v3(input_dim):

    

    # default model - without parameters tuning

    

    K.clear_session()

    x_in = layers.Input(shape=input_dim)

    x = layers.Dense(128, activation='relu')(x_in)

    x = layers.Dropout(.25)(x)

    x = layers.Dense(128, activation='relu')(x)

    x = layers.Dropout(.25)(x)

    x_out = layers.Dense(2, activation=None, name='pred')(x)   

    

    m = models.Model(inputs=x_in, outputs=x_out, name='NeuralNet')

    m.compile(optimizer=Adam(lr=.0005), loss=nll_gaussian_loss, metrics=[laplace_log_likelihood])

    

    return m



def create_model_v4(input_dim, params):

    

    # for parameters tuning

    # params: num_layers, num_units, learning_rate, dropout_rate, activation



    n_units = params['num_units']

    n_layers = params['num_layers']

    dropout_rate = params['dropout_rate']

    activation = params['activation']

    lr = params['learning_rate']

    

    K.clear_session()

    x_in = layers.Input(shape=input_dim)

    x = x_in

    for _ in range(n_layers):

        x = layers.Dense(n_units, activation=activation)(x)

        x = layers.Dropout(dropout_rate)(x)

    x_out = layers.Dense(2, activation=None, name='pred')(x)   

    m = models.Model(inputs=x_in, outputs=x_out, name='NeuralNet')

    m.compile(optimizer=Adam(lr=lr), loss=nll_gaussian_loss, metrics=[laplace_log_likelihood])

    return m
def train_model_cv(train_df, train_params, test_df=None, nfolds=NFOLDS):

    '''

    return: train_preds, oof_preds, [test_preds], trained_models, training_history

    '''

    cat_cols = ['SmokingStatus']

    num_cols = ['base_Weeks', 'Weeks_passed', 'Age']# 'ref_FVC']

    pass_cols = ['base_Percent', 'Sex']

    all_cols = cat_cols + num_cols + pass_cols



    target_cols = ['target_ratio', 'base_FVC', 'weights']





    X = train_df[all_cols].copy()

    y = train_df[target_cols]

    group_train = train_df.Patient.values

    

    if test_df is not None:

        X_test = test_df[all_cols].copy()

        test_fvc_baseline = test_df['base_FVC'].values

        test_preds = np.zeros(shape=(len(X_test), 2))

    

    transformer = ColumnTransformer([

            ('cat',OneHotEncoder(),cat_cols),

            ('num',MinMaxScaler(), num_cols)

        ], remainder='passthrough')

    

    oof_preds = pd.DataFrame(np.zeros(shape=(len(X), 2)), index=X.index, columns=['FVC', 'Confidence'])

    tr_preds = pd.DataFrame(np.zeros(shape=(len(X), 2)), index=X.index, columns=['FVC', 'Confidence'])

    trained_models = dict()

    histories = dict()



    cv = GroupKFold(n_splits=nfolds)

    pbar = tqdm(desc='Group K-folds', total=nfolds)

    for i, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=group_train), start=1):

        X_tr = X.iloc[tr_idx]

        y_tr = y.iloc[tr_idx]

        X_val = X.iloc[val_idx]

        y_val = y.iloc[val_idx]



        X_tr_trans = transformer.fit_transform(X_tr)

        X_val_trans = transformer.transform(X_val)



        neuralnet = create_model_v4(input_dim=X_tr_trans.shape[1], params=train_params)



        hx = neuralnet.fit(X_tr_trans, y_tr, 

                       batch_size=128, 

                       epochs=5000, 

                       validation_data=(X_val_trans, y_val), 

                       verbose=0,

                       callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=1000, mode='min', restore_best_weights=True)]

                    )



        trained_models[f'cv{i}'] = neuralnet

        histories[f'cv{i}'] = hx

        

        

        

        tr_pred = neuralnet.predict(X_tr_trans)

        tr_pred[:, 0] *= y_tr.iloc[:, 1].values

        tr_pred[:, 1] = np.exp(tr_pred[:, 1])



        oof_pred = neuralnet.predict(X_val_trans)

        oof_pred[:, 0] *= y_val.iloc[:, 1].values

        oof_pred[:, 1] = np.exp(oof_pred[:, 1])



        oof_preds.iloc[val_idx, :] = oof_pred

        tr_preds.iloc[tr_idx, :] += tr_pred

        

        if test_df is not None:

            X_test_trans = transformer.transform(X_test)

            # pred=>[target_ratio, log_sigma]

            test_pred = neuralnet.predict(X_test_trans)

            # convert to predict FVC and sigma

            test_pred[:, 0] *= test_fvc_baseline

            test_pred[:, 1] = np.exp(test_pred[:, 1])

            test_preds += test_pred

        

        pbar.update(1)

    pbar.close()

    

    if test_df is not None:

        test_preds /= nfolds

        test_preds = pd.DataFrame(test_preds, index=test_df.index)

    else:

        test_preds = None

        

    tr_preds /= (nfolds - 1)

    tr_preds['FVC_true'] = (y.target_ratio * y.base_FVC).values #'target_ratio', 'base_FVC'

    oof_preds['FVC_true'] = (y.target_ratio * y.base_FVC).values

    

    return tr_preds, oof_preds, test_preds, trained_models, histories







import optuna



class TuningObjective:

    

    def __init__(self, train_df, nfolds):

        self.train_df = train_df

        self.nfolds = nfolds

        

    def __call__(self, trial: optuna.Trial):

        

        tr_params = {

            'num_units': trial.suggest_int('num_units', 10, 256),

            'num_layers': trial.suggest_int('num_layers', 1, 3),

            'dropout_rate': trial.suggest_uniform('dropout_rate', 0., .9),

            'activation': trial.suggest_categorical('activation', choices=['elu','relu','selu','tanh']),

            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2) 

        }

        

        try:

            tr_preds, oof_preds, test_preds, trained_models, histories = train_model_cv(self.train_df, tr_params, test_df=None, nfolds=self.nfolds)



            score = compute_score(oof_preds.FVC_true, oof_preds.FVC, oof_preds.Confidence, return_vec=True) # the higher the better

            score = np.mean(score[self.train_df.last_3_visits.values])   # only last 3 visits



            return -score  # minimize -> the lower the better

        except:

            return np.nan
if TUNING:

    study = optuna.create_study()

    study.optimize(TuningObjective(train_df, NFOLDS), n_trials=20)

    study_df = study.trials_dataframe()

    print(f"best params:\n{study.best_params}\nbest score:{study.best_value:.4f}")

    study_df.to_csv(os.path.join(WORKING_DIR, 'study_df.csv'), index=False)
best_params = {

    'num_units': 137,

    'num_layers': 2,

    'dropout_rate': 0.28,

    'activation': 'tanh',

    'learning_rate': 0.0055

}



best_params = {

    'num_units': 147,

     'num_layers': 1,

     'dropout_rate': 0.12,

     'activation': 'elu',

     'learning_rate': 0.00014

}

'''

best params:

{'num_units': 256, 'num_layers': 2, 'dropout_rate': 0.01815267204890613, 'activation': 'elu', 'learning_rate': 4.7424362225054834e-05}

best score:7.0147

'''

best_params = {'num_units': 256, 'num_layers': 2, 'dropout_rate': 0.018, 'activation': 'elu', 'learning_rate': 4.7e-05}

#study.best_params
tr_preds, oof_preds, test_preds, cv_models, histories = train_model_cv(train_df, best_params, test_df, nfolds=NFOLDS)
tr_preds.head()
oof_preds.head()
test_preds.head()
def plot_history(hx):

    fig, ax = plt.subplots(ncols=2, figsize=(15, 6))



    xs = range(1, len(hx['loss'])+1)



    pd.Series(hx['loss']).rolling(window=10).mean().plot(ax=ax[0], label='tr', alpha=.7)

    pd.Series(hx['val_loss']).rolling(window=10).mean().plot(ax=ax[0], label='val', alpha=.7)

    ax[0].set_xlabel('epoch')

    ax[0].set_ylabel('loss')

    ax[0].set_ylim(5, 12)

    ax[0].legend()



    pd.Series(hx['laplace_log_likelihood']).rolling(window=10).mean().plot(ax=ax[1], label='tr', alpha=.7)

    pd.Series(hx['val_laplace_log_likelihood']).rolling(window=10).mean().plot(ax=ax[1], label='val', alpha=.7)

    ax[1].set_xlabel('epoch')

    ax[1].set_ylabel('score')

    ax[1].legend()

    ax[1].set_ylim(-11, -6)

    plt.show()

    

def print_result(oof_preds, train_df):

    tmp = oof_preds.copy()

    X = train_df

    tmp['predicted_Weeks'] = (X.base_Weeks + X.Weeks_passed).values

    tmp['Sex'] = X.Sex.values

    tmp['SmokingStatus'] = X.SmokingStatus.values

    tmp['Age'] = X.Age.values

    tmp['last_3_visits'] = X.last_3_visits.values

    tmp['Patient'] = X.Patient.values

    

    # oof-score

    # ---------

    fvc_true = tmp[tmp.last_3_visits].FVC_true

    fvc_pred = tmp[tmp.last_3_visits].FVC

    conf = tmp[tmp.last_3_visits].Confidence

    print(f'oof-score: only last 3 visits = {compute_score(fvc_true, fvc_pred, conf):.5f}')





    fvc_true = tmp.FVC_true

    fvc_pred = tmp.FVC

    sigma = tmp.Confidence



    metric = compute_score(fvc_true, fvc_pred, sigma, return_vec=True)

    tmp['oof_score'] = metric

    metric_mean = np.mean(metric)

    print("oof-score: all data {:.5f}".format(metric_mean))

    

    

    # train-score:

    # -----------

    print()

    fvc_true = tr_preds[tmp.last_3_visits].FVC_true

    fvc_pred = tr_preds[tmp.last_3_visits].FVC

    conf = tr_preds[tmp.last_3_visits].Confidence

    print(f'tr-score: only last 3 visits = {compute_score(fvc_true, fvc_pred, conf):.5f}')

    print("tr-score: all data {:.5f}".format(compute_score(tr_preds.FVC_true, tr_preds.FVC, tr_preds.Confidence)))
hx = histories['cv1'].history 

plot_history(hx)
hx = histories['cv2'].history 

plot_history(hx)
hx = histories['cv3'].history 

plot_history(hx)
hx = histories['cv4'].history 

plot_history(hx)
hx = histories['cv5'].history 

plot_history(hx)
oof_preds.head()
tmp = oof_preds.copy()

X = train_df

tmp['predicted_Weeks'] = (X.base_Weeks + X.Weeks_passed).values

tmp['Sex'] = X.Sex.values

tmp['SmokingStatus'] = X.SmokingStatus.values

tmp['Age'] = X.Age.values

tmp['last_3_visits'] = X.last_3_visits.values

tmp['Patient'] = X.Patient.values
tmp[tmp.Patient=='ID00061637202188184085559']
print_result(oof_preds, train_df)
plt.figure(figsize=(5,5))

ax = sns.scatterplot(x='FVC', y='FVC_true', data=tmp, ax=plt.gca())

ax.plot([1000, 6000], [1000, 6000], linestyle='--', color='r')
err = tmp.FVC_true - tmp.FVC



plt.figure(figsize=(10, 6))

ax = err.reset_index().plot.scatter(x='index', y=0, ax=plt.gca())

ax.axhline(xmax=len(err), linestyle='--', color='k', linewidth=2)

ax.set_ylabel('Error')

plt.show()
err[err < -1000]
train_df.loc[1146]
train_df.loc[200]
ax = sns.distplot(tmp.FVC, label='pred')

ax = sns.distplot(tmp.FVC_true, label='true', ax=ax)

ax.legend()
i = 1

fig = plt.figure(figsize=(20, 10))

for s in tmp.Sex.unique():

    for smoke in tmp.SmokingStatus.unique():

        df = tmp[(tmp.Sex==s)&(tmp.SmokingStatus==smoke)]

        plt.subplot(2, 3, i)

        ax = sns.distplot(df.FVC, label='pred', ax=plt.gca())

        ax = sns.distplot(df.FVC_true, label='true', ax=ax)

        ax.legend()

        ax.set_title(f"Sex: {s}, SmokingStatus: {smoke}")

        i += 1

        

fig.tight_layout()
sns.lmplot(x='predicted_Weeks', y='Confidence', data=tmp, col='SmokingStatus', hue='Sex')
tmp.columns
# score statistics by patient backgrounds

tmp['oof_score'] = compute_score(tmp.FVC_true, tmp.FVC, tmp.Confidence, return_vec=True)



(tmp.groupby(['Sex', 'SmokingStatus'])['oof_score']

     .agg(['mean', 'std', 'median'])

     .reset_index()

     .sort_values('mean', ascending=False)

)
# shap values
test_preds.sample(10)
submit_df = submit_df.merge(test_preds, left_on='Patient_Week', right_index=True)

submit_df = submit_df.drop(columns=['FVC', 'Confidence'])

submit_df.columns = ['Patient_Week', 'FVC', 'Confidence']

submit_df.to_csv('submission.csv', index=False)

submit_df.head()