import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

import lightgbm as lgb

import os

from sklearn.model_selection import KFold

from tqdm.notebook import tqdm

import seaborn as sbn

from  datetime import datetime, timedelta

from sklearn import datasets, linear_model

import random

import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
path = Path('/kaggle/input/osic-pulmonary-fibrosis-progression')

assert path.exists()
model_path = Path('/kaggle/working/model')

if os.path.isdir(model_path) == False:

    os.makedirs(model_path)

assert model_path.exists()
TRAIN_TYPES={"Patient": "category", 

         "Weeks": "int16", "FVC": "int32", 'Percent': 'float32', "Age": "uint8",

        "Sex": "category", "SmokingStatus": "category" }

SUBMISSION_TYPES={"Patient_Week": "category", "FVC": "int32", "Confidence": "int16"}



def read_data(path):

    train_df = pd.read_csv(path/'train.csv', dtype = TRAIN_TYPES)

    test_df = pd.read_csv(path/'test.csv', dtype = TRAIN_TYPES)

    submission_df = pd.read_csv(path/'sample_submission.csv', dtype = SUBMISSION_TYPES)

    train_df.drop_duplicates(keep='first', inplace=True, subset=['Patient','Weeks'])

    return train_df, test_df, submission_df
train_df, test_df, submission_df = read_data(path)
def prepare_submission(df, test_df):

    df['Patient'] = df['Patient_Week'].apply(lambda x:x.split('_')[0])

    df['Weeks'] = df['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

    df = df[['Patient','Weeks','Confidence','Patient_Week']]

    df = df.merge(test_df.drop('Weeks', axis=1).copy(), on=['Patient'])

    return df
submission_df = prepare_submission(submission_df, test_df)
submission_df[((submission_df['Patient'] == 'ID00419637202311204720264') & (submission_df['Weeks'] == 6))].head(5)
def adapt_percent_in_submission():

    previous_match = None

    for i, r in submission_df.iterrows():

        in_training = train_df[(train_df['Patient'] == r['Patient']) & (train_df['Weeks'] == r['Weeks'])]

        if(len(in_training) > 0):

            previous_match = in_training['Percent'].item()

            submission_df.iloc[i, submission_df.columns.get_loc('Percent')] = previous_match

        elif previous_match is not None:

            submission_df.iloc[i, submission_df.columns.get_loc('Percent')] = previous_match
adapt_percent_in_submission()
train_df['WHERE'] = 'train'

test_df['WHERE'] = 'val'

submission_df['WHERE'] = 'test'

data = train_df.append([test_df, submission_df])
data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
base = data.loc[data.Weeks == data.min_week]
sbn.countplot(base['Sex'])
base = base[['Patient','FVC', 'Percent']].copy()

base.columns = ['Patient','min_FVC', 'min_Percent']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base
data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']

data['base_week'] = data['base_week']

del base
data[data['Patient'] == 'ID00421637202311550012437']
COLS = ['Sex','SmokingStatus']

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)
data = data.rename(columns={"Age": "age", "min_FVC": "BASE", "base_week": "week", "Percent": "percent"})

FE += ['age','week','BASE', 'percent']

FE
train_df = data.loc[data.WHERE=='train']

test_df = data.loc[data.WHERE=='val']

submission_df = data.loc[data.WHERE=='test']

del data
train_df.sort_values(['Patient', 'Weeks'], inplace=True)
X = train_df[FE]

X.head(15)
y = train_df['FVC']

y
def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
seed_everything(42)
C1_val = 70

C2_val = 1000

C1, C2 = C1_val, C2_val

q = np.array([0.2, 0.50, 0.8])



def score(y_true, y_pred):

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    #sigma_clip = sigma + C1

    sigma_clip = np.max(sigma)

    delta = np.abs(y_true[:, 0] - fvc_pred)

    delta = np.min(delta)

    sq2 = np.sqrt(2.)

    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)

    return np.mean(metric)



def qloss(y_true, y_pred):

    print('y_true.shape', y_true.shape)

    print('y_pred.shape', y_pred.shape)

    # Pinball loss for multiple quantiles

    # τ relu(y-ŷ) + (1-τ) relu(ŷ-y)

    # q * relu(y_true-y_pred) + (1-q) * relu(y_pred-y_true)

    # alt_loss = (q * F.relu(y_true-y_pred) + (1-q) * F.relu(y_pred-y_true))

    e = y_true - y_pred

    v = np.max(q*e, (q-1)*e)

    return np.mean(v)



def mloss(_lambda):

    def loss(y_true, y_pred):

        y_true = y_true.reshape(-1, 1)

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss


n_estimators = 30000

    

if not 'sub_row' in locals():

    sub_row = 0.75

    

if not 'bagging_freq' in locals():

    bagging_freq = 1

    

if not 'learning_rate' in locals():

    learning_rate = 0.4



leave_size = 4



lgb_params = {

    "objective": 'quantile',

    'n_jobs': 1,

    'max_depth': leave_size + 1,

    'num_leaves': 2**leave_size-1,

    "min_data_in_leaf": 2**(leave_size + 1)-1,

#     'subsample': 0.9,

    "n_estimators": n_estimators,

    'learning_rate': 8e-3,

    'colsample_bytree': 0.9,

    'boosting_type': 'gbdt',

    "early_stopping_rounds": 100,

    'verbosity': 1000,

    "metric": ["rmse", "mse"]

}
cat_feats = ['Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes']
NFOLD = 5

kf = KFold(n_splits=NFOLD, shuffle=False)

pred = {a: np.zeros((train_df.shape[0])) for a in q.tolist()}
ensemble_weights = [2./3, 1./3]

assert np.sum(ensemble_weights) == 1.0
models = []

linear_models = []

for cnt, (tr_idx, val_idx) in tqdm(enumerate(kf.split(X)), total=NFOLD):

    X_train, y_train = X.loc[tr_idx], y.loc[tr_idx]

    X_valid, y_valid = X.loc[val_idx], y.loc[val_idx]

    print(f"FOLD {cnt}", X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

    lin_model = linear_model.Ridge(alpha=1.0)

    lin_model.fit(X=X_train, y=y_train)

    linear_models.append(lin_model)

    for qi, quantile_alpha in enumerate(q.tolist()):

        lgb_params['alpha'] = quantile_alpha

        m_lgb_regressor = lgb.LGBMRegressor(**lgb_params)

        m_lgb_regressor.fit(X=X_train, y=y_train, 

                  eval_set=[(X_train, y_train), (X_valid, y_valid)],

                  eval_names=['train mloss', 'valid mloss'], 

                  eval_metric=lgb_params['metric'],

                  verbose=lgb_params['verbosity'],

                  early_stopping_rounds=lgb_params["early_stopping_rounds"],

                  categorical_feature=cat_feats)

        lin_predict = lin_model.predict(X_valid)

        lgb_predict = m_lgb_regressor.predict(X_valid)

        pred[quantile_alpha][val_idx] = np.average([lin_predict, lgb_predict], axis = 0, weights=ensemble_weights)

        models.append(m_lgb_regressor)
full_preds = np.vstack([pred[a] for a in q]).T

score(np.array(y).reshape(-1, 1), full_preds)
pred = []

for i in range(NFOLD):

    cur_pred = []

    for j, _ in enumerate(q):

        model_idx = i * 3 + j

        model = models[model_idx]

        lin_predict = linear_models[i].predict(submission_df[FE])

        dbmc_predict = model.predict(submission_df[FE])

        cur_pred.append(np.average([lin_predict, dbmc_predict], axis = 0, weights=ensemble_weights))

    pred.append(np.array(cur_pred).T)
preds_array = np.array(pred)

preds_array.shape
final_preds = np.mean(preds_array, axis=0)

final_preds.shape
submission_df['FVC1'] = final_preds[:,1]

submission_df['Confidence1'] = final_preds[:, 2] - final_preds[:, 0]
submission_df.loc[~submission_df.FVC1.isnull(),'FVC'] = submission_df.loc[~submission_df.FVC1.isnull(),'FVC1']

submission_df.loc[~submission_df.FVC1.isnull(),'Confidence'] = submission_df.loc[~submission_df.FVC1.isnull(),'Confidence1']
submission_df['Confidence'] = np.clip(submission_df['Confidence'], a_min=200, a_max=1000)

submission_df['Confidence'].describe()
submission_df[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)
submission_final_df = pd.read_csv("submission.csv")
submission_final_df
submission_final_df.describe().T
for p in test_df['Patient'].unique():

    submission_final_df[submission_final_df['Patient_Week'].str.find(p) == 0]['FVC'].plot()
for p in test_df['Patient'].unique():

    fig, ax = plt.subplots()

    submission_final_df[submission_final_df['Patient_Week'].str.find(p) == 0]['FVC'].plot(ax=ax)