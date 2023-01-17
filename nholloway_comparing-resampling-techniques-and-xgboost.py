import time

import math

import logging 

import random



import pandas as pd

import numpy as np

import scipy as sci

from imblearn import under_sampling, over_sampling, combine

from sklearn.decomposition import PCA

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



import catboost as cb

import xgboost as xgb

import seaborn as sns

from scipy.stats import spearmanr

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from hyperopt import hp, tpe, Trials, STATUS_OK

from hyperopt import fmin



from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, StratifiedKFold

from sklearn.preprocessing import LabelEncoder



from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight') 

%matplotlib inline
df = pd.read_csv('../input/train.csv', low_memory=True)

count_yes = len(df[df.target == 1])

count_no = len(df[df.target== 0])



plt.bar(['No', 'Yes'], [count_no, count_yes])

plt.title('Santander Customer Transaction Prediction')

plt.xlabel('Whether Successful Transaction')

plt.ylabel('Count of Customers')

plt.show()
description = pd.DataFrame(index=['observations(rows)', 'percent missing', 'dtype', 'range'])

numerical = []

categorical = []

# Construct a dataframe of Santander metadata

for col in df.columns:

    obs = df[col].size

    p_nan = round(df[col].isna().sum()/obs, 2)

    num_nan = f'{p_nan}% ({df[col].isna().sum()}/{obs})'

    dtype = 'categorical' if df[col].dtype == object else 'numerical'

    numerical.append(col) if dtype == 'numerical' else categorical.append(col)

    rng = f'{len(df[col].unique())} labels' if dtype == 'categorical' else f'{df[col].min()}-{df[col].max()}'

    description[col] = [obs, num_nan, dtype, rng]



final_results = pd.DataFrame(columns = ['parameters', 'training auc score',

                                       'precision', 'training time', 'parameter tuning time'])



pd.set_option('display.max_columns', 150)

display(description)

display(df.head())
sample = df.sample(n=100)

sample = sample.drop(columns=['ID_code'])

class_1 = len(sample[sample.target == 1])

class_0 = len(sample[sample.target== 0])



plt.bar(['Zero', 'One'], [class_0, class_1])

plt.title('Size 100 Sample Distribution')

plt.xlabel('Target Class Label')

plt.ylabel('Count of Customers')

plt.show()

print(f'Zero: {class_0} \nOne: {class_1}')
y = sample.target

x = sample.drop(columns=['target'])

rus = under_sampling.RandomUnderSampler(random_state=0)

resamp_x, resamp_y= rus.fit_resample(x, y)

# Transform the resampled data into principal components

pca = PCA(n_components=2)

resamp = pd.DataFrame(np.hstack((np.vstack(resamp_y), resamp_x)))



resamp_0 = resamp[resamp[0] == 0.0]

resamp_1 = resamp[resamp[0] == 1.0]

orig_0 = sample[sample.target == 0]

orig_1 = sample[sample.target == 1]



orig_no = pca.fit_transform(orig_0)

orig_yes = pca.fit_transform(orig_1)

resamp_no = pca.fit_transform(resamp_0)

resamp_yes = pca.fit_transform(resamp_1)



ono_x = orig_no[:, 0]

ono_y = orig_no[:, 1]

oyes_x = orig_yes[:, 0]

oyes_y = orig_yes[:, 1]

rno_x = resamp_no[:, 0]

rno_y = resamp_no[:, 1]

ryes_x = resamp_yes[:, 0]

ryes_y = resamp_yes[:, 1]



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs= axs.flatten()

axs[0].set_title('Original Data')

axs[0].scatter(ono_x, ono_y, label='Original Class0')

axs[0].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].set_title('Undersampled Majority Class')

axs[1].scatter(ono_x, ono_y, label='Original Class0')

axs[1].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].set_title('More Balanced Data')

axs[2].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].scatter(oyes_x, oyes_y, label='Original Class1')

axs[0].legend()

axs[1].legend()

axs[2].legend()

fig.delaxes(axs[3])

plt.show()
y = sample.target

x = sample.drop(columns=['target'])

tom = under_sampling.TomekLinks(random_state=0)

resamp_x, resamp_y= tom.fit_resample(x, y)

# Transform the resampled data into principal components

pca = PCA(n_components=2)

resamp = pd.DataFrame(np.hstack((np.vstack(resamp_y), resamp_x)))



resamp_0 = resamp[resamp[0] == 0.0]

resamp_1 = resamp[resamp[0] == 1.0]

orig_0 = sample[sample.target == 0]

orig_1 = sample[sample.target == 1]



orig_no = pca.fit_transform(orig_0)

orig_yes = pca.fit_transform(orig_1)

resamp_no = pca.fit_transform(resamp_0)

resamp_yes = pca.fit_transform(resamp_1)



ono_x = orig_no[:, 0]

ono_y = orig_no[:, 1]

oyes_x = orig_yes[:, 0]

oyes_y = orig_yes[:, 1]

rno_x = resamp_no[:, 0]

rno_y = resamp_no[:, 1]

ryes_x = resamp_yes[:, 0]

ryes_y = resamp_yes[:, 1]



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs= axs.flatten()

axs[0].set_title('Original Data')

axs[0].scatter(ono_x, ono_y, label='Original Class0')

axs[0].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].set_title('Undersampled Majority Class')

axs[1].scatter(ono_x, ono_y, label='Original Class0')

axs[1].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].set_title('More Balanced Data')

axs[2].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].scatter(oyes_x, oyes_y, label='Original Class1')

axs[0].legend()

axs[1].legend()

axs[2].legend()

fig.delaxes(axs[3])

plt.show()
y = sample.target

x = sample.drop(columns=['target'])

aknn = under_sampling.AllKNN(random_state=0, n_neighbors=5)

resamp_x, resamp_y= aknn.fit_resample(x, y)

# Transform the resampled data into principal components

pca = PCA(n_components=2)

resamp = pd.DataFrame(np.hstack((np.vstack(resamp_y), resamp_x)))



resamp_0 = resamp[resamp[0] == 0.0]

resamp_1 = resamp[resamp[0] == 1.0]

orig_0 = sample[sample.target == 0]

orig_1 = sample[sample.target == 1]



orig_no = pca.fit_transform(orig_0)

orig_yes = pca.fit_transform(orig_1)

resamp_no = pca.fit_transform(resamp_0)

resamp_yes = pca.fit_transform(resamp_1)



ono_x = orig_no[:, 0]

ono_y = orig_no[:, 1]

oyes_x = orig_yes[:, 0]

oyes_y = orig_yes[:, 1]

rno_x = resamp_no[:, 0]

rno_y = resamp_no[:, 1]

ryes_x = resamp_yes[:, 0]

ryes_y = resamp_yes[:, 1]



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs= axs.flatten()

axs[0].set_title('Original Data')

axs[0].scatter(ono_x, ono_y, label='Original Class0')

axs[0].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].set_title('Undersampled Majority Class')

axs[1].scatter(ono_x, ono_y, label='Original Class0')

axs[1].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].set_title('More Balanced Data')

axs[2].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].scatter(oyes_x, oyes_y, label='Original Class1')

axs[0].legend()

axs[1].legend()

axs[2].legend()

fig.delaxes(axs[3])

plt.show()
y = sample.target

x = sample.drop(columns=['target'])

enn = under_sampling.EditedNearestNeighbours(random_state=0, n_neighbors=3)

resamp_x, resamp_y= enn.fit_resample(x, y)

# Transform the resampled data into principal components

pca = PCA(n_components=2)

resamp = pd.DataFrame(np.hstack((np.vstack(resamp_y), resamp_x)))



resamp_0 = resamp[resamp[0] == 0.0]

resamp_1 = resamp[resamp[0] == 1.0]

orig_0 = sample[sample.target == 0]

orig_1 = sample[sample.target == 1]



orig_no = pca.fit_transform(orig_0)

orig_yes = pca.fit_transform(orig_1)

resamp_no = pca.fit_transform(resamp_0)

resamp_yes = pca.fit_transform(resamp_1)



ono_x = orig_no[:, 0]

ono_y = orig_no[:, 1]

oyes_x = orig_yes[:, 0]

oyes_y = orig_yes[:, 1]

rno_x = resamp_no[:, 0]

rno_y = resamp_no[:, 1]

ryes_x = resamp_yes[:, 0]

ryes_y = resamp_yes[:, 1]



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs= axs.flatten()

axs[0].set_title('Original Data')

axs[0].scatter(ono_x, ono_y, label='Original Class0')

axs[0].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].set_title('Undersampled Majority Class')

axs[1].scatter(ono_x, ono_y, label='Original Class0')

axs[1].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].set_title('More Balanced Data')

axs[2].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].scatter(oyes_x, oyes_y, label='Original Class1')

axs[0].legend()

axs[1].legend()

axs[2].legend()

fig.delaxes(axs[3])

plt.show()
y = sample.target

x = sample.drop(columns=['target'])

ros = over_sampling.RandomOverSampler(random_state=0, ratio=0.5)

resamp_x, resamp_y= ros.fit_resample(x, y)

# Transform the resampled data into principal components

pca = PCA(n_components=2)

resamp = pd.DataFrame(np.hstack((np.vstack(resamp_y), resamp_x)))



resamp_0 = resamp[resamp[0] == 0.0]

resamp_1 = resamp[resamp[0] == 1.0]

orig_0 = sample[sample.target == 0]

orig_1 = sample[sample.target == 1]



orig_no = pca.fit_transform(orig_0)

orig_yes = pca.fit_transform(orig_1)

resamp_no = pca.fit_transform(resamp_0)

resamp_yes = pca.fit_transform(resamp_1)



ono_x = orig_no[:, 0]

ono_y = orig_no[:, 1]

oyes_x = orig_yes[:, 0]

oyes_y = orig_yes[:, 1]

rno_x = resamp_no[:, 0]

rno_y = resamp_no[:, 1]

ryes_x = resamp_yes[:, 0]

ryes_y = resamp_yes[:, 1]



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs= axs.flatten()

axs[0].set_title('Original Data')

axs[0].scatter(ono_x, ono_y, label='Original Class0')

axs[0].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].set_title('Oversampled Minority Class')

axs[1].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[2].set_title('More Balanced Data')

axs[2].scatter(ono_x, ono_y, label='Original Class0')

axs[2].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[0].legend()

axs[1].legend()

axs[2].legend()

fig.delaxes(axs[3])

plt.show()
y = sample.target

x = sample.drop(columns=['target'])

ada = over_sampling.ADASYN(random_state=0, ratio=0.5)

resamp_x, resamp_y= ada.fit_resample(x, y)

# Transform the resampled data into principal components

pca = PCA(n_components=2)

resamp = pd.DataFrame(np.hstack((np.vstack(resamp_y), resamp_x)))



resamp_0 = resamp[resamp[0] == 0.0]

resamp_1 = resamp[resamp[0] == 1.0]

orig_0 = sample[sample.target == 0]

orig_1 = sample[sample.target == 1]



orig_no = pca.fit_transform(orig_0)

orig_yes = pca.fit_transform(orig_1)

resamp_no = pca.fit_transform(resamp_0)

resamp_yes = pca.fit_transform(resamp_1)



ono_x = orig_no[:, 0]

ono_y = orig_no[:, 1]

oyes_x = orig_yes[:, 0]

oyes_y = orig_yes[:, 1]

rno_x = resamp_no[:, 0]

rno_y = resamp_no[:, 1]

ryes_x = resamp_yes[:, 0]

ryes_y = resamp_yes[:, 1]



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs= axs.flatten()

axs[0].set_title('Original Data')

axs[0].scatter(ono_x, ono_y, label='Original Class0')

axs[0].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].set_title('Oversampled Minority Class')

axs[1].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[2].set_title('More Balanced Data')

axs[2].scatter(ono_x, ono_y, label='Original Class0')

axs[2].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[0].legend()

axs[1].legend()

axs[2].legend()

fig.delaxes(axs[3])   

plt.show()
y = sample.target

x = sample.drop(columns=['target'])

smo = over_sampling.SMOTE(random_state=0, ratio=0.5)

resamp_x, resamp_y= smo.fit_resample(x, y)

# Transform the resampled data into principal components

pca = PCA(n_components=2)

resamp = pd.DataFrame(np.hstack((np.vstack(resamp_y), resamp_x)))



resamp_0 = resamp[resamp[0] == 0.0]

resamp_1 = resamp[resamp[0] == 1.0]

orig_0 = sample[sample.target == 0]

orig_1 = sample[sample.target == 1]



orig_no = pca.fit_transform(orig_0)

orig_yes = pca.fit_transform(orig_1)

resamp_no = pca.fit_transform(resamp_0)

resamp_yes = pca.fit_transform(resamp_1)



ono_x = orig_no[:, 0]

ono_y = orig_no[:, 1]

oyes_x = orig_yes[:, 0]

oyes_y = orig_yes[:, 1]

rno_x = resamp_no[:, 0]

rno_y = resamp_no[:, 1]

ryes_x = resamp_yes[:, 0]

ryes_y = resamp_yes[:, 1]



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs= axs.flatten()

axs[0].set_title('Original Data')

axs[0].scatter(ono_x, ono_y, label='Original Class0')

axs[0].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].set_title('Oversampled Minority Class')

axs[1].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[2].set_title('More Balanced Data')

axs[2].scatter(ono_x, ono_y, label='Original Class0')

axs[2].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[0].legend()

axs[1].legend()

axs[2].legend()

fig.delaxes(axs[3])   

plt.show()
y = sample.target

x = sample.drop(columns=['target'])

smotom = combine.SMOTETomek(random_state=0, ratio=0.5)

resamp_x, resamp_y= smotom.fit_resample(x, y)

# Transform the resampled data into principal components

pca = PCA(n_components=2)

resamp = pd.DataFrame(np.hstack((np.vstack(resamp_y), resamp_x)))



resamp_0 = resamp[resamp[0] == 0.0]

resamp_1 = resamp[resamp[0] == 1.0]

orig_0 = sample[sample.target == 0]

orig_1 = sample[sample.target == 1]



orig_no = pca.fit_transform(orig_0)

orig_yes = pca.fit_transform(orig_1)

resamp_no = pca.fit_transform(resamp_0)

resamp_yes = pca.fit_transform(resamp_1)



ono_x = orig_no[:, 0]

ono_y = orig_no[:, 1]

oyes_x = orig_yes[:, 0]

oyes_y = orig_yes[:, 1]

rno_x = resamp_no[:, 0]

rno_y = resamp_no[:, 1]

ryes_x = resamp_yes[:, 0]

ryes_y = resamp_yes[:, 1]



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs= axs.flatten()

axs[0].set_title('Original Data')

axs[0].scatter(ono_x, ono_y, label='Original Class0')

axs[0].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].set_title('Oversampled Minority Class')

axs[1].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[3].set_title('Undersampled Majority Class')

axs[3].scatter(ono_x, ono_y, label='Original Class0')

axs[3].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].set_title('More Balanced Data')

axs[2].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[0].legend()

axs[1].legend()

axs[2].legend()

axs[3].legend()

  

plt.show()
y = sample.target

x = sample.drop(columns=['target'])

smotenn = combine.SMOTEENN(random_state=0, ratio=0.5)

resamp_x, resamp_y= smotenn.fit_resample(x, y)

# Transform the resampled data into principal components

pca = PCA(n_components=2)

resamp = pd.DataFrame(np.hstack((np.vstack(resamp_y), resamp_x)))



resamp_0 = resamp[resamp[0] == 0.0]

resamp_1 = resamp[resamp[0] == 1.0]

orig_0 = sample[sample.target == 0]

orig_1 = sample[sample.target == 1]



orig_no = pca.fit_transform(orig_0)

orig_yes = pca.fit_transform(orig_1)

resamp_no = pca.fit_transform(resamp_0)

resamp_yes = pca.fit_transform(resamp_1)



ono_x = orig_no[:, 0]

ono_y = orig_no[:, 1]

oyes_x = orig_yes[:, 0]

oyes_y = orig_yes[:, 1]

rno_x = resamp_no[:, 0]

rno_y = resamp_no[:, 1]

ryes_x = resamp_yes[:, 0]

ryes_y = resamp_yes[:, 1]



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs= axs.flatten()

axs[0].set_title('Original Data')

axs[0].scatter(ono_x, ono_y, label='Original Class0')

axs[0].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].set_title('Oversampled Minority Class')

axs[1].scatter(oyes_x, oyes_y, label='Original Class1')

axs[1].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[3].set_title('Undersampled Majority Class')

axs[3].scatter(ono_x, ono_y, label='Original Class0')

axs[3].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].set_title('More Balanced Data')

axs[2].scatter(rno_x, rno_y, label='Undersampled Class0')

axs[2].scatter(ryes_x, ryes_y, label='Oversampled Class1')

axs[0].legend()

axs[1].legend()

axs[2].legend()

axs[3].legend()

  

plt.show()
# Organizes XGB results and extracts metadata from Trials object

def org_results(trials, hyperparams, ratio, model_name):

    fit_idx = -1

    for idx, fit  in enumerate(trials):

        hyp = fit['misc']['vals']

        xgb_hyp = {key:[val] for key, val in hyperparams.items()}

        if hyp == xgb_hyp:

            fit_idx = idx

            break

            

    train_time = str(trials[-1]['refresh_time'] - trials[0]['book_time'])

    acc = round(trials[fit_idx]['result']['accuracy'], 3)

    train_auc = round(trials[fit_idx]['result']['train auc'], 3)

    test_auc = round(trials[fit_idx]['result']['test auc'], 3)

    conf_matrix = trials[fit_idx]['result']['conf matrix']



    results = {

        'model': model_name,

        'ratio': ratio,

        'parameter search time': train_time,

        'accuracy': acc,

        'test auc score': test_auc,

        'training auc score': train_auc,

        'confusion matrix': conf_matrix,

        'parameters': hyperparams

    }

    return results



def data_ratio(y):

    unique, count = np.unique(y, return_counts=True)

    ratio = round(count[0]/count[1], 2)

    return f'{ratio}:1 ({count[0]}/{count[1]})'
batch_size = 10000

xgb_df = df.sample(batch_size)

y = xgb_df['target'].reset_index(drop=True)

x = xgb_df.drop(columns=['target','ID_code'])

smotomek = combine.SMOTETomek(random_state=0, ratio=0.5)

bal_x, bal_y= smotomek.fit_resample(x, y)



samp_len = len(bal_y)

xgb_df2 = df.sample(samp_len - batch_size)

xgb_df = pd.concat([xgb_df, xgb_df2])

imb_y = xgb_df['target'].reset_index(drop=True)

imb_x = xgb_df.drop(columns=['target','ID_code'])
def xgb_train(data_x, data_y, md_name):

    ratio = data_ratio(data_y)

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.20)

   

    def xgb_objective(space, early_stopping_rounds=50):



        model = XGBClassifier(

            learning_rate = space['learning_rate'], 

            n_estimators = int(space['n_estimators']), 

            max_depth = int(space['max_depth']), 

            min_child_weight = space['m_child_weight'], 

            gamma = space['gamma'], 

            subsample = space['subsample'], 

            colsample_bytree = space['colsample_bytree'],

            objective = 'binary:logistic'

        )



        model.fit(train_x, train_y, 

                  eval_set = [(train_x, train_y), (test_x, test_y)],

                  eval_metric = 'auc',

                  early_stopping_rounds = early_stopping_rounds,

                  verbose = False)



        predictions = model.predict(test_x)

        test_preds = model.predict_proba(test_x)[:,1]

        train_preds = model.predict_proba(train_x)[:,1]



        xgb_booster = model.get_booster()

        train_auc = roc_auc_score(train_y, train_preds)

        test_auc = roc_auc_score(test_y, test_preds)

        accuracy = accuracy_score(test_y, predictions) 

        conf_matrix = confusion_matrix(test_y, predictions)



        return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,

                'test auc': test_auc, 'train auc': train_auc, 'conf matrix': conf_matrix

               }



    space = {

        'n_estimators': hp.quniform('n_estimators', 50, 1000, 25),

        'max_depth': hp.quniform('max_depth', 1, 12, 1),

        'm_child_weight': hp.quniform('m_child_weight', 1, 6, 1),

        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),

        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),

        'learning_rate': hp.loguniform('learning_rate', np.log(.001), np.log(.3)),

        'colsample_bytree': hp.quniform('colsample_bytree', .5, 1, .1)

    }



    trials = Trials()

    xgb_hyperparams = fmin(fn = xgb_objective, 

                     max_evals = 25, 

                     trials = trials,

                     algo = tpe.suggest,

                     space = space

                     )

    

    results = org_results(trials.trials, xgb_hyperparams, ratio, md_name)

    return results



bal_results = xgb_train(bal_x, bal_y, 'Balanced Data')

imb_results = xgb_train(imb_x, imb_y, 'Imbalanced Data')
bal_confusion = bal_results.pop('confusion matrix')

imb_confusion = imb_results.pop('confusion matrix')
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.heatmap(bal_confusion, annot=True, cmap= 'viridis_r', ax=ax[0])

sns.heatmap(imb_confusion, annot=True, cmap= 'viridis_r', ax=ax[1])

ax[0].set_title('Balanced Dataset')

ax[1].set_title('Imbalanced Dataset')

plt.show()

final_results = pd.DataFrame([bal_results, imb_results])

display(final_results) 