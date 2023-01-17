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
train = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/training_set_features.csv")

train_label = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/training_set_labels.csv")

test = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/test_set_features.csv")

submission = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/submission_format.csv")
test
train
train_dataset = train.merge(train_label, on='respondent_id')

test_dataset = test.merge(submission, on='respondent_id')
train_test = pd.concat([train_dataset,test_dataset])

train_test
common = set(train.columns) & set(test.columns)

train_diff = set(train.columns) - common

test_diff = set(test.columns) - common
train_diff, test_diff
train_label
submission
train.info()
non_number_columns = train.dtypes[train.dtypes == object].index.values

non_number_columns
train['age_group'].value_counts()
train['education'].value_counts()
train['race'].value_counts()
train['income_poverty'].value_counts()
non_number_columns
error = ['education', 'income_poverty', 'marital_status', 'rent_or_own', 'employment_status', 'employment_industry', 'employment_occupation']

success = set(non_number_columns) - set(error)
for column in error:

    train[column] = train[column].astype(str)

train
train['income_poverty'].value_counts()
from sklearn.preprocessing import LabelEncoder



def column_label_LE(data,columns):

    dataset = pd.DataFrame()

    dataset['Label'] = data[columns]

    

    le = LabelEncoder()

    dataset['Number'] = le.fit_transform(data[columns])

    dataset['Type'] = columns

    return dataset.drop_duplicates()

pd.concat([column_label_LE(train,columns) for columns in success])
pd.concat([column_label_LE(train,columns) for columns in error])
for column in error:

    train_test[column] = train_test[column].astype(str)



for columns in non_number_columns:

    le = LabelEncoder()

    train_test[columns] = le.fit_transform(train_test[columns])



dataset = pd.concat([column_label_LE(train_test,columns) for columns in non_number_columns])

dataset
train_test
len(train), len(test)
len(train_test[len(train):])
x_train = train_test.iloc[:len(train)*9//10].drop(['respondent_id','seasonal_vaccine'], axis=1)

x_val = train_test.iloc[len(train)*9//10:].drop(['respondent_id','seasonal_vaccine'], axis=1)

x_test = train_test.iloc[len(train):].drop(['respondent_id','seasonal_vaccine'], axis=1)



y_train = train_test.iloc[:len(train)*9//10]['seasonal_vaccine']

y_val = train_test.iloc[len(train)*9//10:]['seasonal_vaccine']
import time

from xgboost import XGBRegressor

ts = time.time()



model = XGBRegressor(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 20)



time.time() - ts
Y_pred = model.predict(x_val).clip(0, 20)

Y_test = model.predict(x_test)
Y_test
submission['seasonal_vaccine'] = Y_test

submission

submission.to_csv('submission_xgb.csv',index=False)
submission.to_csv('submission_xgb.csv',index=False)
dataset
train
train_label
# train_label['h1n1_vaccine'].value_counts()
train_dataset = train.merge(train_label, on='respondent_id')

train_dataset
train_dataset.info()
sample = dataset[dataset['Type']=='age_group']

sample
train['education'].value_counts()
train['age_group'].value_counts()
test
import seaborn as sns

import matplotlib.pyplot as plt



for columns in train_dataset.columns[1:]:

    print(columns)

    print('Mean',test_dataset[columns].mean())

    print('Median',test_dataset[columns].median())

    sns.distplot(test_dataset[columns]) # , kde=False, fit=stats.gamma

    plt.show()
train_label
train_label['h1n1_vaccine'].value_counts()
import seaborn as sns

sns.distplot(train_label['seasonal_vaccine'])
submission