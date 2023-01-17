# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Loading the CSV Files

train = pd.read_csv('/kaggle/input/machine-learning-in-agriculture/train_yaOffsB.csv')

test = pd.read_csv('/kaggle/input/machine-learning-in-agriculture/test_pFkWwen.csv')

sub_sample = pd.read_csv('/kaggle/input/machine-learning-in-agriculture/sample_submission_O1oDc4H.csv')
import lightgbm as lgb

from matplotlib import pyplot as plt

from sklearn import preprocessing

from sklearn.metrics import mean_squared_log_error, mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import classification_report

import seaborn as sns
train.head()
train.shape
test.head()
test.shape
train['Crop_Damage'].value_counts()
fig, ax = plt.subplots()

# plot histogram

ax.hist(train['Crop_Damage'])

# set title and labels

ax.set_xticks((0,1,2))

ax.set_title('Crop Damage')

ax.set_xlabel('Outcome')

ax.set_ylabel('Counts')

plt.show()
fig, ax = plt.subplots()

# plot histogram

ax.hist(train['Crop_Type'])

# set title and labels

ax.set_xticks((0,1))

ax.set_title('Crop_Type')

ax.set_xlabel('Types')

ax.set_ylabel('Counts')

plt.show()
#Concatenating the Datasets

train['train_flag'] = 1

test['train_flag'] = 0

test['Crop_Damage'] = 0





data = pd.concat((train, test))

data.shape
#select relevant features from data

feature_cols = train.columns.tolist()

feature_cols.remove('ID')

feature_cols.remove('Crop_Damage')

feature_cols.remove('train_flag')

label_col = 'Crop_Damage'

print(feature_cols)
#ID values are actually integers with some prefixes. Convert them back to int.

data['ID_value'] = data['ID'].apply(lambda x: x.strip('F')).astype('int')
data = data.sort_values(['ID_value'])
data.head()
data = data.reset_index(drop=True)
data.head()
data['Soil_Type_Damage'] = data.sort_values(['ID_value']).groupby(['Soil_Type'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values



data['Estimated_Insects_Count_Damage'] = data.sort_values(['ID_value']).groupby(['Estimated_Insects_Count'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values



data['Crop_Type_Damage'] = data.sort_values(['ID_value']).groupby(['Crop_Type'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values



data['Pesticide_Use_Category_Damage'] = data.sort_values(['ID_value']).groupby(['Pesticide_Use_Category'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values



data['Season_Damage'] = data.sort_values(['ID_value']).groupby(['Season'])['Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values



data['Soil_Type_Damage_lag2'] = data.sort_values(['ID_value']).groupby(['Soil_Type'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values



data['Estimated_Insects_Count_Damage_lag2'] = data.sort_values(['ID_value']).groupby(['Estimated_Insects_Count'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values



data['Crop_Type_Damage_lag2'] = data.sort_values(['ID_value']).groupby(['Crop_Type'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values



data['Pesticide_Use_Category_Damage_lag2'] = data.sort_values(['ID_value']).groupby(['Pesticide_Use_Category'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values



data['Season_Damage_lag2'] = data.sort_values(['ID_value']).groupby(['Season'])['Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

data.loc[data['train_flag'] == 0, 'Crop_Damage'] = -999
data['Crop_Damage_lag1'] = data['Crop_Damage'].shift(fill_value=-999)

data['Estimated_Insects_Count_lag1'] = data['Estimated_Insects_Count'].shift(fill_value=-999)

data['Crop_Type_lag1'] = data['Crop_Type'].shift(fill_value=-999)

data['Soil_Type_lag1'] = data['Soil_Type'].shift(fill_value=-999)

data['Pesticide_Use_Category_lag1'] = data['Pesticide_Use_Category'].shift(fill_value=-999)

data['Number_Doses_Week_lag1'] = data['Number_Doses_Week'].shift(fill_value=-999)

data['Number_Weeks_Used_lag1'] = data['Number_Weeks_Used'].shift(fill_value=-999)

data['Number_Weeks_Quit_lag1'] = data['Number_Weeks_Quit'].shift(fill_value=-999)

data['Season_lag1'] = data['Season'].shift(fill_value=-999)



data['Crop_Damage_lag2'] = data['Crop_Damage'].shift(periods=2,fill_value=-999)

data['Estimated_Insects_Count_lag2'] = data['Estimated_Insects_Count'].shift(periods=2,fill_value=-999)

data['Crop_Type_lag2'] = data['Crop_Type'].shift(fill_value=-999)

data['Soil_Type_lag2'] = data['Soil_Type'].shift(fill_value=-999)

data['Pesticide_Use_Category_lag2'] = data['Pesticide_Use_Category'].shift(periods=2,fill_value=-999)

data['Number_Doses_Week_lag2'] = data['Number_Doses_Week'].shift(periods=2,fill_value=-999)

data['Number_Weeks_Used_lag2'] = data['Number_Weeks_Used'].shift(periods=2,fill_value=-999)

data['Number_Weeks_Quit_lag2'] = data['Number_Weeks_Quit'].shift(periods=2,fill_value=-999)

data['Season_lag2'] = data['Season'].shift(periods=2,fill_value=-999)
train, test = data[data.train_flag == 1], data[data.train_flag == 0]
train.drop(['train_flag'], inplace=True, axis=1)

test.drop(['train_flag'], inplace=True, axis=1)

test.drop([label_col], inplace=True, axis=1);
missing_impute = -999

train['Number_Weeks_Used'] = train['Number_Weeks_Used'].apply(lambda x: missing_impute if pd.isna(x) else x)

test['Number_Weeks_Used'] = test['Number_Weeks_Used'].apply(lambda x: missing_impute if pd.isna(x) else x)



train['Number_Weeks_Used_lag1'] = train['Number_Weeks_Used_lag1'].apply(lambda x: missing_impute if pd.isna(x) else x)

test['Number_Weeks_Used_lag1'] = test['Number_Weeks_Used_lag1'].apply(lambda x: missing_impute if pd.isna(x) else x)



train['Number_Weeks_Used_lag2'] = train['Number_Weeks_Used_lag2'].apply(lambda x: missing_impute if pd.isna(x) else x)

test['Number_Weeks_Used_lag2'] = test['Number_Weeks_Used_lag2'].apply(lambda x: missing_impute if pd.isna(x) else x);
#Split the datasets into training and evaluation sets

df_train, df_eval = train_test_split(train, test_size=0.40, random_state=42, shuffle=True, stratify=train[label_col])
feature_cols = train.columns.tolist()

feature_cols.remove('ID')

feature_cols.remove('Crop_Damage')

feature_cols.remove('ID_value')

label_col = 'Crop_Damage'

print(feature_cols)
train.head()
#Categorical Columns

cat_cols = ['Crop_Type', 'Soil_Type', 'Pesticide_Use_Category', 'Season', 'Crop_Type_lag1', 'Soil_Type_lag1', 'Pesticide_Use_Category_lag1', 'Season_lag1']
#light GBM Parameters

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
clf = lgb.LGBMClassifier(**params)

    

clf.fit(df_train[feature_cols], df_train[label_col], early_stopping_rounds=100, eval_set=[(df_train[feature_cols], df_train[label_col]), (df_eval[feature_cols], df_eval[label_col])], eval_metric='multi_error', verbose=True, categorical_feature=cat_cols)



eval_score = accuracy_score(df_eval[label_col], clf.predict(df_eval[feature_cols]))



print('Eval ACC: {}'.format(eval_score))
best_iter = clf.best_iteration_

params['n_estimators'] = best_iter

print(params)
df_train = pd.concat((df_train, df_eval))
clf = lgb.LGBMClassifier(**params)



clf.fit(df_train[feature_cols], df_train[label_col], eval_metric='multi_error', verbose=False, categorical_feature=cat_cols)



# eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))

eval_score_acc = accuracy_score(df_train[label_col], clf.predict(df_train[feature_cols]))



print('ACC: {}'.format(eval_score_acc))
preds = clf.predict(test[feature_cols])
submission = pd.DataFrame({'ID':test['ID'], 'Crop_Damage':preds})

submission.to_csv('Submission.csv',index=False)