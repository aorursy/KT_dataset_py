# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/mlsp-hackathon/train_HK6lq50.csv')
test = pd.read_csv('../input/mlsp-hackathon/test_wF0Ps6O.csv')
train.shape
test.shape
train.head(3)
train.dtypes
train.nunique()
data = pd.concat([train,test], axis=0, sort= False)
data.shape
data.isnull().sum()
conti = ['test_id','age','total_programs_enrolled']
for col in conti:
    sns.distplot((data[col]), bins=100)
    plt.show()
categ = data.columns.drop(['id', 'trainee_id','age','test_id'])
for cols in categ:
    sns.countplot(data[cols])
    plt.xticks(rotation = 90)
    plt.show()
for cols in categ:
    plt.figure(figsize=(20,4))
    sns.countplot(x= data[cols] , hue= data['trainee_engagement_rating'])
    plt.show()
data['age'].fillna(data['age'].mean(), inplace = True)
def impute_rating(cols):
    rate = cols[0]
    tp = cols[1]
    if pd.isnull(rate):
        if tp=='Y':
            return 4
        else:
            return 1
    else:
        return rate      
data['trainee_engagement_rating'] = data[['trainee_engagement_rating','program_type']].apply(impute_rating, axis=1)
data.isnull().sum()
data.head()
#splitting data
df_train = data.iloc[0:73147]
df_test = data.iloc[73147:]
X = df_train.drop('is_pass', axis=1)
y = df_train['is_pass']
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)
X_test = df_test.drop('is_pass', axis=1)
def sub_type(col):
    return col.split("_")[1]
X_train_featured = X_train.copy()
X_train_featured['sub_type']  = X_train_featured['program_id'].apply(sub_type)

X_valid_featured = X_valid.copy()
X_valid_featured['sub_type']  = X_valid_featured['program_id'].apply(sub_type)

X_test_featured = X_test.copy()
X_test_featured['sub_type']  = X_test_featured['program_id'].apply(sub_type)
X_train_featured['test_type'].replace({'online':1,'offline':2}, inplace = True)
X_train_featured['gender'].replace({'M':1,'F':0}, inplace = True)
X_train_featured['is_handicapped'].replace({'Y':1,'N':0}, inplace = True)
X_train_featured['education'].replace({'High School Diploma':1,
                                       'Matriculation':2,
                                       'Bachelors':3,
                                       'No Qualification':4,
                                       'Masters':4 }, inplace = True)
X_train_featured['difficulty_level'].replace({'easy':1,
                                       'intermediate':2,
                                       'hard':3,
                                       'vary hard':4}, inplace = True)

X_valid_featured['test_type'].replace({'online':1,'offline':2}, inplace = True)
X_valid_featured['gender'].replace({'M':1,'F':0}, inplace = True)
X_valid_featured['is_handicapped'].replace({'Y':1,'N':0}, inplace = True)
X_valid_featured['education'].replace({'High School Diploma':1,
                                       'Matriculation':2,
                                       'Bachelors':3,
                                       'No Qualification':4,
                                       'Masters':4 }, inplace = True)
X_valid_featured['difficulty_level'].replace({'easy':1,
                                       'intermediate':2,
                                       'hard':3,
                                       'vary hard':4}, inplace = True)

X_test_featured['test_type'].replace({'online':1,'offline':2}, inplace = True)
X_test_featured['gender'].replace({'M':1,'F':0}, inplace = True)
X_test_featured['is_handicapped'].replace({'Y':1,'N':0}, inplace = True)
X_test_featured['education'].replace({'High School Diploma':1,
                                       'Matriculation':2,
                                       'Bachelors':3,
                                       'No Qualification':4,
                                       'Masters':4 }, inplace = True)
X_test_featured['difficulty_level'].replace({'easy':1,
                                       'intermediate':2,
                                       'hard':3,
                                       'vary hard':4}, inplace = True)
from category_encoders import CountEncoder
ce = CountEncoder()
ce.fit(X_train_featured['program_type'])
X_train_featured['program_type'+'_count'] = ce.transform(X_train_featured['program_type'])
X_valid_featured['program_type'+'_count'] = ce.transform(X_valid_featured['program_type'])
X_test_featured['program_type'+'_count'] = ce.transform(X_test_featured['program_type'])
X_train_featured.drop(['id','program_id','program_type'], axis=1, inplace = True)
X_test_featured.drop(['id','program_id','program_type'], axis=1, inplace = True)
X_valid_featured.drop(['id','program_id','program_type'], axis=1, inplace = True)
X_train_featured.drop('trainee_id', axis=1, inplace = True)
X_test_featured.drop('trainee_id', axis=1, inplace = True)
X_valid_featured.drop('trainee_id', axis=1, inplace = True)
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
scaler.fit(X_train_featured)

X_train_scaled = pd.DataFrame(scaler.transform(X_train_featured),columns=X_train_featured.columns)
X_valid_scaled = pd.DataFrame(scaler.transform(X_valid_featured),columns=X_valid_featured.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_featured),columns=X_test_featured.columns)
from sklearn.metrics import roc_auc_score,classification_report
X_test_scaled.shape
plt.figure(figsize=(12,8))
sns.heatmap(X_train_scaled.corr(), annot=True, cmap = 'viridis')
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train_scaled,y_train)
pred = model_lr.predict(X_valid_scaled)
print('roc_auc_score:')
print(roc_auc_score(y_valid,pred))
print('classification_report:')
print(classification_report(y_valid,pred))
from sklearn.ensemble import RandomForestClassifier
model_lr = RandomForestClassifier()
model_lr.fit(X_train_scaled,y_train)
pred = model_lr.predict(X_valid_scaled)
print('roc_auc_score:')
print(roc_auc_score(y_valid,pred))
print('classification_report:')
print(classification_report(y_valid,pred))
from xgboost import XGBClassifier
model_lr = XGBClassifier()
model_lr.fit(X_train_scaled,y_train)
pred = model_lr.predict(X_valid_scaled)
print('roc_auc_score:')
print(roc_auc_score(y_valid,pred))
print('classification_report:')
print(classification_report(y_valid,pred))
estimate = range(100,1000,100)
tune = {}
from sklearn.ensemble import RandomForestClassifier
for n in estimate:
    model_lr = RandomForestClassifier(n_estimators=n)
    model_lr.fit(X_train_scaled,y_train)
    pred = model_lr.predict(X_valid_scaled)
    tune[n] = roc_auc_score(y_valid,pred)
keys = list(tune.keys())
values = list(tune.values())
plt.figure(figsize=(12,6))
plt.xlabel('n_estimators')
sns.lineplot(keys, values)
estimate = range(1,202,50)
tune2 = {}
from sklearn.ensemble import RandomForestClassifier
for n in estimate:
    model_lr = RandomForestClassifier(n_estimators=300, max_depth=n)
    model_lr.fit(X_train_scaled,y_train)
    pred = model_lr.predict(X_valid_scaled)
    tune2[n] = roc_auc_score(y_valid,pred)
    
keys = list(tune2.keys())
values = list(tune2.values())
plt.figure(figsize=(12,6))
sns.lineplot(keys, values)
plt.xlabel('max_depth')
estimate = range(20,222,50)
tune2 = {}
from sklearn.ensemble import RandomForestClassifier
for n in estimate:
    model_lr = RandomForestClassifier(n_estimators=300, max_depth=50,max_leaf_nodes=n)
    model_lr.fit(X_train_scaled,y_train)
    pred = model_lr.predict(X_valid_scaled)
    tune2[n] = roc_auc_score(y_valid,pred)
    
keys = list(tune2.keys())
values = list(tune2.values())
plt.figure(figsize=(12,6))
sns.lineplot(keys, values)
plt.xlabel('max_leaf_nodes')
my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train_scaled, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid_scaled, y_valid)], 
             verbose=False)
pred = my_model.predict(X_valid_scaled)
print(roc_auc_score(y_valid, pred))
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, max_depth=50)
model.fit(X_train_scaled,y_train)
sub = test['id']
sub.head()
pred = pd.DataFrame(model.predict(X_test_scaled),columns=['is_pass'])
pred.head()
sub = pd.concat([sub,pred], axis=1)
sub.head()
sub.to_csv('submissionD.csv', index=0)

sub.shape
train_X = pd.concat([X_train_scaled,X_valid_scaled],axis=0)
train_y = pd.concat([y_train,y_valid])
train_X.shape
model = RandomForestClassifier(n_estimators=300, max_depth=50)
model.fit(train_X, train_y)
