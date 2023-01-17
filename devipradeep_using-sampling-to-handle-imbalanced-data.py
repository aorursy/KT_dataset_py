!pip install missingpy

!pip install imblearn
import pandas as pd

import numpy as np

import os, shutil

import pprint

from collections import Counter

import joblib



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import recall_score

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC



from xgboost import XGBClassifier

from missingpy import MissForest

from imblearn import over_sampling, under_sampling



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
!pwd
DATA_PATH = '/kaggle/input/novartis-data'



train = pd.read_csv(os.path.join(DATA_PATH, 'Train.csv'))

test_df = pd.read_csv(os.path.join(DATA_PATH, 'Test.csv'))

print("We have {} rows and {} columns".format(train.shape[0], train.shape[1]))


# Target value distribution

target_vc = train.MULTIPLE_OFFENSE.value_counts()



sns.barplot(x = target_vc.index, y = target_vc.values)

plt.show()
train.isna().sum()
imputer = MissForest()

imputed_data = imputer.fit_transform(train.drop(['INCIDENT_ID', 'DATE', 'MULTIPLE_OFFENSE'], axis=1))

train['X_12'] = imputed_data[:, 11]

del imputed_data
train.drop(['INCIDENT_ID', 'DATE'], axis=1, inplace=True)
# Function for balancing the data



def data_sampling(df, over_sampling_strategy, under_sampling_strategy, target_column='MULTIPLE_OFFENSE'):

  #Over sampling and undersampling funcitons

  over = over_sampling.SMOTE(sampling_strategy=over_sampling_strategy)

  under = under_sampling.RandomUnderSampler(sampling_strategy=under_sampling_strategy)



  over_sampled_data, _ = over.fit_resample(train.values, train[target_column].values)

  sampled_data, _ = under.fit_resample(over_sampled_data, over_sampled_data[:, 15])

  #Converting sampled data to pandas dataframe

  sampled_df = pd.DataFrame(sampled_data)

  sampled_df.columns = train.columns

  return sampled_df
def create_folds(df, n_folds, target_column='MULTIPLE_OFFENSE'):

  df['kFold'] = -1

  kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=18)

  for fold, (train_idxs,val_idxs) in enumerate(kfold.split(X=df, y=df[target_column].values)):

    df.loc[val_idxs, 'kFold'] = fold

  return df
def train_fn(df, n_folds, save_model=True, booster = 'gbtree', learning_rate = 0.1, max_depth = 3, subsample = 1, target_column='MULTIPLE_OFFENSE'):

  all_recalls = []

  if save_model:

    if not os.path.exists('/kaggle/working/models'):

      os.mkdir('/kaggle/working/models')

    else:

      shutil.rmtree('/kaggle/working/models')

      os.mkdir('/kaggle/working/models')

  for fold in range(n_folds):

    train_df = df[df.kFold.isin(FOLD_MAPPING.get(fold))]

    val_df = df[df.kFold == fold]



    train_X = train_df.drop([target_column, 'kFold'], axis=1)

    train_y = train_df[target_column].values



    val_X = val_df.drop([target_column, 'kFold'], axis=1)

    val_y = val_df[target_column].values



    #Model

    clf = XGBClassifier(booster=booster, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample)

    clf.fit(train_X, train_y)

    predictions = clf.predict(val_X)

    if save_model:

      joblib.dump(clf, f"models/{algo}_{fold}.pkl")

      print(f"Saved {algo}_{fold}.pkl")

    recall = recall_score(val_y, predictions)

    all_recalls.append(recall)

    # print("Recall score  for fold {} is {}".format(fold, recall_score(val_y, predictions)))

  return all_recalls, np.mean(all_recalls)
def test(test_df):

  test_idxs = test_df.INCIDENT_ID.values

  imputer = MissForest()

  imputed_data = imputer.fit_transform(test_df.drop(['INCIDENT_ID', 'DATE'], axis=1))

  test_df['X_12'] = imputed_data[:, 11]

  del imputed_data

  test_df = test_df.drop(['INCIDENT_ID', 'DATE'], axis=1)

  predictions = pd.DataFrame()

  for fold in range(n_folds):

    clf = joblib.load(f"/kaggle/working/models/{algo}_{fold}.pkl")

    predictions[f"pred_{fold}"] = clf.predict(test_df)

  final_predictions = predictions.mode(axis=1)[0].values

  submission = pd.DataFrame({'INCIDENT_ID':test_idxs, 'MULTIPLE_OFFENSE':final_predictions})

  return submission
n_folds = 5

FOLD_MAPPING = {

    0 : [1,2,3,4],

    1 : [0,2,3,4],

    2 : [1,0,3,4],

    3 : [1,2,0,4],

    4 : [1,2,3,0],

}



parameters = {

    'over_sampling_strategy' : np.arange(0.1, 0.6, 0.1),

    'under_sampling_strategy' : np.arange(0.8, 0.4, -0.1),

    'booster' : ['gbtree', 'dart'],

    'learning_rate' : np.arange(0.1, 0.4, 0.1),

    'max_depth' : np.arange(3, 8, 1),

    'subsample' : np.arange(0.5, 1.25, 0.25)

}
# scores = []



# for o in parameters['over_sampling_strategy']:

#   for u in parameters['under_sampling_strategy']:

#       sampled_df = data_sampling(train, o, u)

#       df = create_folds(sampled_df, n_folds)

#       recall_arr, recall_mean = train_fn(df, n_folds, False, 'gbtree', 0.1, 6)

#       scores.append((o, u, recall_arr, recall_mean))

#       print(o, u, recall_mean)



# scores = sorted(scores, key=lambda x : x[3], reverse=True)

# scores[0:2]
# o = 0.5

# u = 0.5

# scores2 = []

# for b in parameters['booster']:

#   for lr in parameters['learning_rate']:

#     for d in parameters['max_depth']:

#         sampled_df = data_sampling(train, o, u)

#         df = create_folds(sampled_df, n_folds)

#         recall_arr, recall_mean = train_fn(df, n_folds, False, b, lr, d)

#         scores2.append((b, lr, d, recall_arr, recall_mean))



# scores2 = sorted(scores2, key=lambda x : x[3], reverse=True)

# scores2[0:2]
o = 0.5

u = 0.5

algo = 'xgb_gbtree'

sampled_df = data_sampling(train, o, u)

df = create_folds(sampled_df, n_folds)

recall_arr, recall_mean = train_fn(df, n_folds, True, booster='gbtree', learning_rate=0.2, max_depth=4)

print(recall_mean)

print(recall_arr)
os.listdir('/kaggle/working/models')
submission = test(test_df)

submission.to_csv(f"/kaggle/working/submission.csv", index=False)