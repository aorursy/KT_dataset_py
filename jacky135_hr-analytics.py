# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/wns-inno/train_LZdllcl.csv')

test = pd.read_csv('/kaggle/input/wns-inno/test_2umaH9m.csv')

sample = pd.read_csv('/kaggle/input/wns-inno/sample_submission_M0L0uXE.csv')
test_id = test['employee_id']
train
train.info()
train['previous_year_rating'].fillna(3.0, inplace=True)

train['education'].fillna('Bachelor\'s', inplace=True)

test['previous_year_rating'].fillna(3.0, inplace=True)

test['education'].fillna('Bachelor\'s', inplace=True)
train['is_promoted'].value_counts(normalize = True)
sns.countplot(train['is_promoted'])
plt.figure(figsize=(20,10))

sns.countplot(x='department',hue = "is_promoted",data = train)
plt.figure(figsize=(20,10))

ax = sns.countplot(x='region',hue = "is_promoted",data = train)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
plt.figure(figsize=(20,10))

ax = sns.countplot(x='education',hue = "is_promoted",data = train)
plt.figure(figsize=(20,10))

ax = sns.countplot(x='gender',hue = "is_promoted",data = train)

plt.figure(figsize=(20,10))

ax = sns.countplot(x='recruitment_channel',hue = "is_promoted",data = train)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
train['no_of_trainings'].value_counts()
plt.figure(figsize=(20,10))

ax = sns.countplot(x='no_of_trainings',hue = "is_promoted",data = train)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
plt.figure(figsize=(20,10))

ax = sns.countplot(x='age',hue = "is_promoted",data = train)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
plt.figure(figsize=(20,10))

ax = sns.countplot(x='previous_year_rating',hue = "is_promoted",data = train)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
plt.figure(figsize=(20,10))

ax = sns.countplot(x='length_of_service',hue = "is_promoted",data = train)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
plt.figure(figsize=(20,10))

ax = sns.countplot(x='KPIs_met >80%',hue = "is_promoted",data = train)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
plt.figure(figsize=(20,10))

ax = sns.countplot(x='awards_won?',hue = "is_promoted",data = train)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
plt.figure(figsize=(20,10))

ax = sns.countplot(x='avg_training_score',hue = "is_promoted",data = train)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
def impute_age(col):

    age = col

    if age <= 27:

       return 1.0

    elif age>27 and age <=35:

       return 2.0

    elif age>=36 and age <=39:

       return 3.0

    else :

       return 4.0

    

       
train['age'] = train['age'].apply(impute_age)

test['age'] = test['age'].apply(impute_age)
train
y = train['is_promoted']
train.drop(['is_promoted','employee_id'],axis = 1,inplace = True)

test.drop(['employee_id'],axis = 1,inplace = True)



train['department']= label_encoder.fit_transform(train['department'])

train['region']= label_encoder.fit_transform(train['region'])

train['education']= label_encoder.fit_transform(train['education'])

train['gender']= label_encoder.fit_transform(train['gender'])

train['recruitment_channel']= label_encoder.fit_transform(train['recruitment_channel'])

train['no_of_trainings']=label_encoder.fit_transform(train['no_of_trainings'])

train['age']= label_encoder.fit_transform(train['age'])

train['previous_year_rating']= label_encoder.fit_transform(train['previous_year_rating'])

train['length_of_service']= label_encoder.fit_transform(train['length_of_service'])

train['KPIs_met >80%']= label_encoder.fit_transform(train['KPIs_met >80%'])

train['awards_won?']= label_encoder.fit_transform(train['awards_won?'])

train['avg_training_score']=label_encoder.fit_transform(train['avg_training_score'])


test['department']= label_encoder.fit_transform(test['department'])

test['region']= label_encoder.fit_transform(test['region'])

test['education']= label_encoder.fit_transform(test['education'])

test['gender']= label_encoder.fit_transform(test['gender'])

test['recruitment_channel']= label_encoder.fit_transform(test['recruitment_channel'])

test['no_of_trainings']=label_encoder.fit_transform(test['no_of_trainings'])

test['age']= label_encoder.fit_transform(test['age'])

test['previous_year_rating']= label_encoder.fit_transform(test['previous_year_rating'])

test['length_of_service']= label_encoder.fit_transform(test['length_of_service'])

test['KPIs_met >80%']= label_encoder.fit_transform(test['KPIs_met >80%'])

test['awards_won?']= label_encoder.fit_transform(test['awards_won?'])

test['avg_training_score']=label_encoder.fit_transform(test['avg_training_score'])
train
from sklearn.preprocessing import MinMaxScaler

min_max = MinMaxScaler()

train_minmax  = min_max.fit_transform(train)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_minmax, y, test_size=0.1, random_state=42)
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import f1_score
import xgboost as xgb

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

model = LGBMClassifier(learning_rate=0.09, n_estimators=200, max_depth=4, min_child_weight=6

                      ,nthread=4, subsample=0.8, colsample_bytree=0.6,scale_pos_weight=3,seed=29)



model.fit(X_train, y_train)


y_predict = model.predict(X_test)
precision = precision_score(y_test, y_predict)

recall = recall_score(y_test, y_predict)

f1 = f1_score(y_test, y_predict)
print(precision)
print(recall)
print(f1)
print("roc_auc test set", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
print("roc_auc training set", roc_auc_score(y_train, model.predict_proba(X_train)[:,1]))
test_minmax  = min_max.transform(test)
y_pred = model.predict(test_minmax)
Y_pred = pd.Series(y_pred,name="is_promoted")

submission = pd.concat([pd.Series(test_id,name="employee_id"),Y_pred],axis = 1)
submission.to_csv("HR_Analytics.csv",index=False)
submission['is_promoted'].value_counts()