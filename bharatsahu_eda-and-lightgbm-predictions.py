import os

from os import listdir

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



#plotly

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot

# import cufflinks

# cufflinks.go_offline()

# cufflinks.set_config_file(world_readable=True, theme='pearl')



import seaborn as sns

sns.set(style="whitegrid")



from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

from functools import partial

import random

import math

import scipy



from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.metrics import mean_squared_error

import category_encoders as ce



from PIL import Image

import cv2

import lightgbm as lgb





#pydicom

import pydicom



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')





# Settings for pretty nice plots

plt.style.use('fivethirtyeight')
IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progressiont/"



train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
train_df.head(5)
print("Total number of Patient records in Train set: ",train_df['Patient'].count())

print("Total number of Patient recods in Test set: ",test_df['Patient'].count())
print(f"The total patient records are {train_df['Patient'].count()}, from those the unique patient ids are {train_df['Patient'].value_counts().shape[0]} ")
columns = train_df.keys()

columns = list(columns)

print(columns)
print('Train .dcm number of images:', len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/train'))), '\n' +

      'Test .dcm number of images:', len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/test'))), '\n' +

      '-----------------------', '\n' +

      'There is the same number of images as patients in train/ test .csv datasets')
files = folders = 0



path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"



for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files += len(filenames)

    folders += len(dirnames)



print("{:,} files/images, {:,} folders/patients".format(files, folders))
files = []

for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files.append(len(filenames))



print("{:,} average files/images per patient".format(round(np.mean(files))))

print("{:,} max files/images per patient".format(round(np.max(files))))

print("{:,} min files/images per patient".format(round(np.min(files))))
OUTPUT_DICT = './'



ID = 'Patient_Week'

TARGET = 'FVC'

N_FOLD = 5

train_df[ID] = train_df['Patient'].astype(str) + '_' + train_df['Weeks'].astype(str)

print(train_df.shape)

train_df.head()
output = pd.DataFrame()

gb = train_df.groupby('Patient')

tk0 = tqdm(gb, total=len(gb))

for _, usr_df in tk0:

    usr_output = pd.DataFrame()

    for week, tmp in usr_df.groupby('Weeks'):

        rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'}

        tmp = tmp.drop(columns='Patient_Week').rename(columns=rename_cols)

        drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent']

        _usr_output = usr_df.drop(columns=drop_cols).rename(columns={'Weeks': 'predict_Week'}).merge(tmp, on='Patient')

        _usr_output['Week_passed'] = _usr_output['predict_Week'] - _usr_output['base_Week']

        usr_output = pd.concat([usr_output, _usr_output])

    output = pd.concat([output, usr_output])

    

train_df = output[output['Week_passed']!=0].reset_index(drop=True)

print(train_df.shape)

train_df.head()
# construct test input



test_df = test_df.rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'})

submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split('_')[0])

submission['predict_Week'] = submission['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

test_df = submission.drop(columns=['FVC', 'Confidence']).merge(test_df, on='Patient')

test_df['Week_passed'] = test_df['predict_Week'] - test_df['base_Week']

print(test_df.shape)

test_df.head()
folds = train_df[[ID, 'Patient', TARGET]].copy()

Fold = GroupKFold(n_splits=N_FOLD)

groups = folds['Patient'].values

for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):

    folds.loc[val_index, 'fold'] = int(n)

folds['fold'] = folds['fold'].astype(int)

folds.head()
def get_logger(filename='log'):

    logger = getLogger(__name__)

    logger.setLevel(INFO)

    handler1 = StreamHandler()

    handler1.setFormatter(Formatter("%(message)s"))

    handler2 = FileHandler(filename=f"{filename}.log")

    handler2.setFormatter(Formatter("%(message)s"))

    logger.addHandler(handler1)

    logger.addHandler(handler2)

    return logger



logger = get_logger()





def run_single_lightgbm(param, train_df, test_df, folds, features, target, fold_num=0, categorical=[]):

    

    trn_idx = folds[folds.fold != fold_num].index

    val_idx = folds[folds.fold == fold_num].index

    logger.info(f'len(trn_idx) : {len(trn_idx)}')

    logger.info(f'len(val_idx) : {len(val_idx)}')

    

    if categorical == []:

        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],

                               label=target.iloc[trn_idx])

        val_data = lgb.Dataset(train_df.iloc[val_idx][features],

                               label=target.iloc[val_idx])

    else:

        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],

                               label=target.iloc[trn_idx],

                               categorical_feature=categorical)

        val_data = lgb.Dataset(train_df.iloc[val_idx][features],

                               label=target.iloc[val_idx],

                               categorical_feature=categorical)



    oof = np.zeros(len(train_df))

    predictions = np.zeros(len(test_df))



    num_round = 10000



    clf = lgb.train(param,

                    trn_data,

                    num_round,

                    valid_sets=[trn_data, val_data],

                    verbose_eval=100,

                    early_stopping_rounds=100)



    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)



    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')

    fold_importance_df["fold"] = fold_num



    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration)

    

    # RMSE

    logger.info("fold{} RMSE score: {:<8.5f}".format(fold_num, np.sqrt(mean_squared_error(target[val_idx], oof[val_idx]))))

    

    return oof, predictions, fold_importance_df





def run_kfold_lightgbm(param, train, test, folds, features, target, n_fold=5, categorical=[]):

    

    logger.info(f"================================= {n_fold}fold lightgbm =================================")

    

    oof = np.zeros(len(train))

    predictions = np.zeros(len(test))

    feature_importance_df = pd.DataFrame()



    for fold_ in range(n_fold):

        print("Fold {}".format(fold_))

        _oof, _predictions, fold_importance_df = run_single_lightgbm(param,

                                                                     train,

                                                                     test,

                                                                     folds,

                                                                     features,

                                                                     target,

                                                                     fold_num=fold_,

                                                                     categorical=categorical)

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        oof += _oof

        predictions += _predictions / n_fold



    # RMSE

    logger.info("CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(target, oof))))



    logger.info(f"=========================================================================================")

    

    return feature_importance_df, predictions, oof



    

def show_feature_importance(feature_importance_df, name):

    cols = (feature_importance_df[["Feature", "importance"]]

            .groupby("Feature")

            .mean()

            .sort_values(by="importance", ascending=False)[:50].index)

    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



    plt.figure(figsize=(6, 4))

    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('Features importance (averaged/folds)')

    plt.tight_layout()

    plt.savefig(OUTPUT_DICT+f'feature_importance_{name}.png')
target = train_df[TARGET]

test_df[TARGET] = np.nan



# features

cat_features = ['Sex', 'SmokingStatus']

num_features = [c for c in test_df.columns if (test_df.dtypes[c] != 'object') & (c not in cat_features)]

features = num_features + cat_features

drop_features = [ID, TARGET, 'predict_Week', 'base_Week']

features = [c for c in features if c not in drop_features]



if cat_features:

    ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='impute')

    ce_oe.fit(train_df)

    train_df = ce_oe.transform(train_df)

    test_df = ce_oe.transform(test_df)

        

lgb_param = {'objective': 'regression',

             'metric': 'rmse',

             'boosting_type': 'gbdt',

             'learning_rate': 0.01,

             'seed': 123,

             'max_depth': -1,

             'verbosity': -1,

            }



feature_importance_df, predictions, oof = run_kfold_lightgbm(lgb_param, train_df, test_df, folds, features, target, 

                                                             n_fold=N_FOLD, categorical=cat_features)

    

show_feature_importance(feature_importance_df, TARGET)
train_df['FVC_pred'] = oof

test_df['FVC_pred'] = predictions
# baseline score

train_df['Confidence'] = 100

train_df['sigma_clipped'] = train_df['Confidence'].apply(lambda x: max(x, 70))

train_df['diff'] = abs(train_df['FVC'] - train_df['FVC_pred'])

train_df['delta'] = train_df['diff'].apply(lambda x: min(x, 1000))

train_df['score'] = -math.sqrt(2)*train_df['delta']/train_df['sigma_clipped'] - np.log(math.sqrt(2)*train_df['sigma_clipped'])

score = train_df['score'].mean()

print(score)


def loss_func(weight, row):

    confidence = weight

    sigma_clipped = max(confidence, 70)

    diff = abs(row['FVC'] - row['FVC_pred'])

    delta = min(diff, 1000)

    score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)

    return -score



results = []

tk0 = tqdm(train_df.iterrows(), total=len(train_df))

for _, row in tk0:

    loss_partial = partial(loss_func, row=row)

    weight = [100]

    result = scipy.optimize.minimize(loss_partial, weight, method='SLSQP')

    x = result['x']

    results.append(x[0])
# optimized score

train_df['Confidence'] = results

train_df['sigma_clipped'] = train_df['Confidence'].apply(lambda x: max(x, 70))

train_df['diff'] = abs(train_df['FVC'] - train_df['FVC_pred'])

train_df['delta'] = train_df['diff'].apply(lambda x: min(x, 1000))

train_df['score'] = -math.sqrt(2)*train_df['delta']/train_df['sigma_clipped'] - np.log(math.sqrt(2)*train_df['sigma_clipped'])

score = train_df['score'].mean()

print(score)
TARGET = 'Confidence'



target = train_df[TARGET]

test_df[TARGET] = np.nan



# features

cat_features = ['Sex', 'SmokingStatus']

num_features = [c for c in test_df.columns if (test_df.dtypes[c] != 'object') & (c not in cat_features)]

features = num_features + cat_features

drop_features = [ID, TARGET, 'predict_Week', 'base_Week', 'FVC', 'FVC_pred']

features = [c for c in features if c not in drop_features]



lgb_param = {'objective': 'regression',

             'metric': 'rmse',

             'boosting_type': 'gbdt',

             'learning_rate': 0.01,

             'seed': 123,

             'max_depth': -1,

             'verbosity': -1,

            }



feature_importance_df, predictions, oof = run_kfold_lightgbm(lgb_param, train_df, test_df, folds, features, target, 

                                                             n_fold=N_FOLD, categorical=cat_features)

    

show_feature_importance(feature_importance_df, TARGET)
train_df['Confidence'] = oof

train_df['sigma_clipped'] = train_df['Confidence'].apply(lambda x: max(x, 70))

train_df['diff'] = abs(train_df['FVC'] - train_df['FVC_pred'])

train_df['delta'] = train_df['diff'].apply(lambda x: min(x, 1000))

train_df['score'] = -math.sqrt(2)*train_df['delta']/train_df['sigma_clipped'] - np.log(math.sqrt(2)*train_df['sigma_clipped'])

score = train_df['score'].mean()

print(score)
test_df['Confidence'] = predictions
sub = submission.drop(columns=['FVC', 'Confidence']).merge(test_df[['Patient_Week', 'FVC_pred', 'Confidence']], 

                                                           on='Patient_Week')

sub.columns = submission.columns

sub.to_csv('submission.csv', index=False)

sub.head()