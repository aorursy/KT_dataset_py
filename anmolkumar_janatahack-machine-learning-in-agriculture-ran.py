# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

pd.options.display.max_columns = None

pd.options.display.max_rows = None





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

from matplotlib import *

from matplotlib import pyplot as plt

from catboost import CatBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
train_data = pd.read_csv('/kaggle/input/av-janatahack-machine-learning-in-agriculture/train.csv')

test_data = pd.read_csv('/kaggle/input/av-janatahack-machine-learning-in-agriculture/test.csv')

sample_submission = pd.read_csv('/kaggle/input/av-janatahack-machine-learning-in-agriculture/sample_submission.csv')

train_data.columns = train_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print('Train Data Shape: ', train_data.shape)

print('Test Data Shape: ', test_data.shape)

train_data.head()
train_data.dtypes
train_data.isnull().sum()
train_data.nunique()
print('Season volume:\n', train_data.season.value_counts())

print('Crop volume:\n', train_data.crop_type.value_counts())

print('Soil volume:\n', train_data.soil_type.value_counts())

print('Pesticide volume:\n', train_data.pesticide_use_category.value_counts())

print('Soil wise crop damage volume:\n', train_data.groupby('soil_type')['crop_damage'].value_counts())

print('Crop wise crop damage volume:\n', train_data.groupby('crop_type')['crop_damage'].value_counts())

print('Season wise crop damage volume:\n', train_data.groupby('season')['crop_damage'].value_counts())
plt.figure(figsize = (10, 8))

sns.scatterplot(x = 'number_doses_week', y = 'estimated_insects_count', hue = 'season', data = train_data)

plt.show()
sns.catplot(x = 'pesticide_use_category', y = 'number_weeks_used', data = train_data, hue = 'crop_damage')

sns.despine()
sns.catplot(x = 'crop_type', y = 'estimated_insects_count', data = train_data, hue = 'crop_damage')

sns.despine()
sns.catplot(x = 'crop_type', y = 'number_doses_week', kind = 'violin', data = train_data, hue = 'crop_damage')

sns.despine()
ax = sns.violinplot(x = 'soil_type', y = 'number_doses_week', hue = 'pesticide_use_category', data = train_data)
ax = sns.violinplot(x = 'crop_type', y = 'number_doses_week', hue = 'pesticide_use_category', data = train_data)
ax = sns.violinplot(x = 'season', y = 'number_doses_week', hue = 'pesticide_use_category', data = train_data)
sns.catplot(x = 'crop_damage', y = 'number_weeks_used', data = train_data, hue = 'season')

sns.despine()
sns.catplot(x = "soil_type", y = 'estimated_insects_count', kind = 'box', data = train_data)

sns.despine()
sns.catplot(x = "crop_type", y = 'estimated_insects_count', kind = 'box', data = train_data)

sns.despine()
sns.pairplot(train_data[['estimated_insects_count', 'number_doses_week', 'crop_damage', 'number_weeks_used', 'number_weeks_quit']], hue = 'crop_damage')

sns.despine()
plt.figure(figsize = (16, 5))

for i in range(1, 5):

    cols = ['estimated_insects_count', 'number_doses_week', 'number_weeks_used', 'number_weeks_quit']

    col = cols[i-1]

    ax = plt.subplot(1, 4, i)

    plt.boxplot(train_data.loc[~(train_data[col].isnull()), col], patch_artist = True, widths = 0.6)

    ax.set_title(col)
train_data['id'] = train_data['id'].apply(lambda x: x.split('F')[1])

test_data['id'] = test_data['id'].apply(lambda x: x.split('F')[1])

train_data.loc[(train_data['pesticide_use_category'] == 1), 'number_weeks_used'] = 0
testData = test_data.copy()

testData['crop_damage'] = -1

train_data['flag'] = 'train'

test_data['flag'] = 'test'

#master_data = train_data[train_data.columns[~train_data.columns.isin(['crop_damage'])]].append(test_data[train_data.columns[~train_data.columns.isin(['crop_damage'])]])

master_data = train_data.append(testData)

master_data = master_data.sort_values(by = ['id'], ascending = (True))

master_data = master_data.reset_index(drop = True)
master_data['soil_type_damage'] = master_data.sort_values(['id']).groupby(['soil_type'])['crop_damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['estimated_insects_count_damage'] = master_data.sort_values(['id']).groupby(['estimated_insects_count'])['crop_damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['crop_type_damage'] = master_data.sort_values(['id']).groupby(['crop_type'])['crop_damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['pesticide_use_category_damage'] = master_data.sort_values(['id']).groupby(['pesticide_use_category'])['crop_damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['season_damage'] = master_data.sort_values(['id']).groupby(['season'])['crop_damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['soil_type_damage_lag2'] = master_data.sort_values(['id']).groupby(['soil_type'])['crop_damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['estimated_insects_count_damage_lag2'] = master_data.sort_values(['id']).groupby(['estimated_insects_count'])['crop_damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['crop_type_damage_lag2'] = master_data.sort_values(['id']).groupby(['crop_type'])['crop_damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['pesticide_use_category_damage_lag2'] = master_data.sort_values(['id']).groupby(['pesticide_use_category'])['crop_damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['season_damage_lag2'] = master_data.sort_values(['id']).groupby(['season'])['crop_damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

master_data['crop_damage_lag1'] = master_data['crop_damage'].shift(fill_value=-999)

master_data['estimated_insects_count_lag1'] = master_data['estimated_insects_count'].shift(fill_value=-999)

master_data['crop_type_lag1'] = master_data['crop_type'].shift(fill_value=-999)

master_data['soil_type_lag1'] = master_data['soil_type'].shift(fill_value=-999)

master_data['pesticide_use_category_lag1'] = master_data['pesticide_use_category'].shift(fill_value=-999)

master_data['number_doses_week_lag1'] = master_data['number_doses_week'].shift(fill_value=-999)

master_data['number_weeks_used_lag1'] = master_data['number_weeks_used'].shift(fill_value=-999)

master_data['number_weeks_quit_lag1'] = master_data['number_weeks_quit'].shift(fill_value=-999)

master_data['season_lag1'] = master_data['season'].shift(fill_value=-999)



master_data['crop_damage_lag2'] = master_data['crop_damage'].shift(periods=2,fill_value=-999)

master_data['estimated_insects_count_lag2'] = master_data['estimated_insects_count'].shift(periods=2,fill_value=-999)

master_data['crop_type_lag2'] = master_data['crop_type'].shift(fill_value=-999)

master_data['soil_type_lag2'] = master_data['soil_type'].shift(fill_value=-999)

master_data['pesticide_use_category_lag2'] = master_data['pesticide_use_category'].shift(periods=2,fill_value=-999)

master_data['number_doses_week_lag2'] = master_data['number_doses_week'].shift(periods=2,fill_value=-999)

master_data['number_weeks_used_lag2'] = master_data['number_weeks_used'].shift(periods=2,fill_value=-999)

master_data['number_weeks_quit_lag2'] = master_data['number_weeks_quit'].shift(periods=2,fill_value=-999)

master_data['season_lag2'] = master_data['season'].shift(periods=2,fill_value=-999)
train_data, test_data = master_data[master_data.flag == 'train'], master_data[master_data.flag == 'test']
impute = -999
train_data = train_data.drop(['flag'], axis = 1)

test_data = test_data.drop(['flag'], axis = 1)

test_data = test_data.drop(['crop_damage'], axis = 1)
print(train_data.shape, test_data.shape)
train_data.head()
train_data['number_weeks_used'] = train_data['number_weeks_used'].apply(lambda x: impute if pd.isna(x) else x)

test_data['number_weeks_used'] = test_data['number_weeks_used'].apply(lambda x: impute if pd.isna(x) else x)



train_data['number_weeks_used_lag1'] = train_data['number_weeks_used_lag1'].apply(lambda x: impute if pd.isna(x) else x)

test_data['number_weeks_used_lag1'] = test_data['number_weeks_used_lag1'].apply(lambda x: impute if pd.isna(x) else x)



train_data['number_weeks_used_lag2'] = train_data['number_weeks_used_lag2'].apply(lambda x: impute if pd.isna(x) else x)

test_data['number_weeks_used_lag2'] = test_data['number_weeks_used_lag2'].apply(lambda x: impute if pd.isna(x) else x)
X_train, X_test = train_test_split(train_data, test_size = 0.40, random_state = 22, shuffle = True, stratify = train_data['crop_damage'])
cat_cols = ['crop_type', 'soil_type', 'pesticide_use_category', 'season', 'crop_type_lag1', 'soil_type_lag1', 'pesticide_use_category_lag1', 'season_lag1']

feature_cols = train_data.columns.tolist()

feature_cols.remove('id')

feature_cols.remove('crop_damage')
params = {}

params['learning_rate'] = 0.04

params['max_depth'] = 18

params['n_estimators'] = 3000

params['objective'] = 'multiclass'

params['boosting_type'] = 'gbdt'

params['subsample'] = 0.7

params['random_state'] = 42

params['colsample_bytree']=0.7

params['min_data_in_leaf'] = 55

params['reg_alpha'] = 1.7

params['reg_lambda'] = 1.11

params['class_weight']: {0: 0.44, 1: 0.4, 2: 0.37}
clf = LGBMClassifier(**params)

    

clf.fit(X_train[feature_cols], X_train['crop_damage'], early_stopping_rounds = 100, 

        eval_set = [(X_train[feature_cols], X_train['crop_damage']), (X_test[feature_cols], X_test['crop_damage'])],

        eval_metric='multi_error', verbose = True, categorical_feature = cat_cols)



eval_score = accuracy_score(X_test['crop_damage'], clf.predict(X_test[feature_cols]))
print('Eval ACC: {}'.format(eval_score))
train_data = pd.concat((X_train, X_test))
clf = LGBMClassifier(**params)



clf.fit(X_train[feature_cols], X_train['crop_damage'], eval_metric = 'multi_error', verbose = False, categorical_feature = cat_cols)



# eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))

eval_score_acc = accuracy_score(X_train['crop_damage'], clf.predict(X_train[feature_cols]))



print('ACC: {}'.format(eval_score_acc))
Ypreds = clf.predict(test_data[feature_cols])
submission = pd.DataFrame({'ID': test_data['id'], 'Crop_Damage': Ypreds})

submission.to_csv('agriculture.csv', index = False)