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
train=pd.read_csv('../input/pet-adoption/Dataset/train.csv')
test=pd.read_csv('../input/pet-adoption/Dataset/test.csv')
train.head(5)
train.info()
train.isnull().sum()
print(train['condition'].unique())

print(test['condition'].unique())
train["condition"]=train["condition"].fillna(train['condition'].mean())
test["condition"]=test["condition"].fillna(test['condition'].mean())
train.isnull().sum()
test.isnull().sum()
train['condition'].unique()
test['condition'].unique()
train.info(5)
train['color_type'].unique
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
train['color_type']= label_encoder.fit_transform(train['color_type'])
test['color_type']= label_encoder.fit_transform(test['color_type'])
  
train['color_type'].unique() 

train.info(5)
train.head()
train['issue_date'] = pd.to_datetime(train['issue_date'])
train['listing_date'] = pd.to_datetime(train['listing_date'])

test['issue_date'] = pd.to_datetime(test['issue_date'])
test['listing_date'] = pd.to_datetime(test['listing_date'])
train.head()
test.info()
train['issue_date_year']=train['issue_date'].dt.year
train['issue_date_month']=train['issue_date'].dt.month
train['issue_date_week']=train['issue_date'].dt.week
train['issue_date_day']=train['issue_date'].dt.day
'''train['issue_date_hour']=train['issue_date'].dt.hour
train['issue_date_min']=train['issue_date'].dt.week
train['issue_date_dayofweek']=train['issue_date'].dt.dayofweek'''

test['issue_date_year']=test['issue_date'].dt.year
test['issue_date_month']=test['issue_date'].dt.month
test['issue_date_week']=test['issue_date'].dt.week
test['issue_date_day']=test['issue_date'].dt.day
'''test['issue_date_hour']=test['issue_date'].dt.hour
test['issue_date_min']=test['issue_date'].dt.week
test['issue_date_dayofweek']=test['issue_date'].dt.dayofweek'''

train['listing_date_year']=train['listing_date'].dt.year
train['listing_date_month']=train['listing_date'].dt.month
train['listing_date_week']=train['listing_date'].dt.week
train['listing_date_day']=train['listing_date'].dt.day
'''train['listing_date_hour']=train['listing_date'].dt.hour
train['listing_date_min']=train['listing_date'].dt.week
train['listing_date_dayofweek']=train['listing_date'].dt.dayofweek'''

test['listing_date_year']=test['listing_date'].dt.year
test['listing_date_month']=test['listing_date'].dt.month
test['listing_date_week']=test['listing_date'].dt.week
test['listing_date_day']=test['listing_date'].dt.day
'''test['listing_date_hour']=test['listing_date'].dt.hour
test['listing_date_min']=test['listing_date'].dt.week
test['listing_date_dayofweek']=test['listing_date'].dt.dayofweek'''


train.drop(columns=['issue_date','listing_date'],axis=1,inplace=True)
test.drop(columns=['issue_date','listing_date'],axis=1,inplace=True)


train.info()
train['length(m)']
train.head(5)
train['X1_X2']=train['X1']*train['X2']
test['X1_X2']=test['X1']*test['X2']
train.info()
test.info()
title=train['pet_id']
train.head()
'''id=[]
for i in list(train.pet_id):
    if not i.isdigit():
        id.append(i.replace("ANSL_","").strip().split(" ")[0])
train["pet_id"]=id
'''

'''id=[]
for i in list(test.pet_id):
    if not i.isdigit():
        id.append(i.replace("ANSL_","").strip().split(" ")[0])
test["pet_id"]=id'''
train.head()
test.head()
'''train["pet_id"] = train["pet_id"].astype(str).astype(int)
print(train.dtypes)'''

train["breed_category"] = train["breed_category"].astype(int)
print(train.dtypes)
'''test["pet_id"] = test["pet_id"].astype(str).astype(int)'''
print(test.dtypes)
print(train.dtypes)
train_breed=train['breed_category']
train_pet=train['pet_category']

print(train_breed)
print(train_pet)
train.drop(columns=['breed_category','pet_category',],axis=1,inplace=True)
train1=train.copy()
train2=train.copy()
train1['breed_category']=train_breed
train2['pet_category']=train_pet

train1.info()


from sklearn import preprocessing
test.info()
train2.head(5)
test
train1
train2
train_data = train1.drop('breed_category', axis=1)
target = train1['breed_category']

train_data.drop(columns=['pet_id'],inplace=True)


train_data


'''from xgboost import XGBClassifier
classifier =XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=20, shuffle=True, random_state=0) '''
'''scoring = 'accuracy'
score = cross_val_score(classifier, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)'''
'''score.mean()'''
'''test_data=test'''
'''test_data'''
'''from xgboost import XGBClassifier
classifier =XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)


classifier.fit(train_data,target)
test_data=test.drop('pet_id', axis=1)
prediction=classifier.predict(test_data)'''
'''submission=pd.DataFrame({"pet_id":test["pet_id"],"breed_category":prediction})
submission.to_csv('Submission.csv', index=False)'''
train_data1 = train2.drop('pet_category', axis=1)
target1 = train2['pet_category']


train_data1.drop(columns=['pet_id'],inplace=True)




'''from xgboost import XGBClassifier
classifier =XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)'''
'''scoring = 'accuracy'
scores = cross_val_score(classifier, train_data1, target1, cv=k_fold, n_jobs=1, scoring=scoring)
print(scores)'''
'''scores.mean()'''
'''from xgboost import XGBClassifier
classifier =XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)
classifier.fit(train_data1,target1)
test_data1=test.drop('pet_id', axis=1)

predictions=classifier.predict(test_data1)'''
'''submission1=pd.DataFrame({"pet_id":test["pet_id"],"pet_category":predictions})
submission1.to_csv('Submission1.csv', index=False)'''
'''output1=pd.read_csv('./Submission1.csv')'''
'''output2=pd.read_csv('./Submission.csv')'''
'''output1'''

'''output2'''
'''output2['pet_category']=output1['pet_category']'''
'''output2 '''

'''output2.to_csv('final.csv', index = False)'''
'''output2.info()'''
'''train_data'''

'''target1'''

'''import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import datasets

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions'''
'''
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=lr)'''
'''label = ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']
clf_list = [clf1, clf2, clf3, sclf]
    
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2, 2)
grid = itertools.product([0,1],repeat=2)

clf_cv_mean = []
clf_cv_std = []
#for clf, label, grd in zip(clf_list, label, grid):
        
scores = cross_val_score(clf2, train_data, target, cv=3, scoring='accuracy')
print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
clf_cv_mean.append(scores.mean())
clf_cv_std.append(scores.std())
        
clf2.fit(train_data, target)
 '''
    
'''test_data=test.drop('pet_id', axis=1)
prediction=clf2.predict(test_data)'''
'''submission=pd.DataFrame({"pet_id":test["pet_id"],"breed_category":prediction})
submission.to_csv('Submission.csv', index=False)'''
'''train_data1 = train2.drop('pet_category', axis=1)
target1 = train2['pet_category']


train_data1.drop(columns=['pet_id'],inplace=True)
'''

'''label = ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']
clf_list = [clf1, clf2, clf3, sclf]
    
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2, 2)
grid = itertools.product([0,1],repeat=2)

clf_cv_mean = []
clf_cv_std = []
#for clf, label, grd in zip(clf_list, label, grid):
        
scores = cross_val_score(clf2, train_data1, target1, cv=3, scoring='accuracy')
print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
clf_cv_mean.append(scores.mean())
clf_cv_std.append(scores.std())
        
clf2.fit(train_data1, target1)
 
    '''
'''test_data1=test.drop('pet_id', axis=1)
prediction=clf2.predict(test_data1)'''
'''submission1=pd.DataFrame({"pet_id":test["pet_id"],"pet_category":predictions})
submission1.to_csv('Submission1.csv', index=False)'''
'''output1=pd.read_csv('./Submission1.csv')'''
'''output2=pd.read_csv('./Submission.csv')'''

'''output1'''
'''output2'''
'''output2['pet_category']=output1['pet_category']'''
'''output2'''
'''output2.to_csv('final.csv', index = False)'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=15)
decision_tree.fit(train_data,target)
test_data=test.drop('pet_id', axis=1)

predictions=decision_tree.predict(test_data)
submission=pd.DataFrame({"pet_id":test["pet_id"],"breed_category":predictions})
submission.to_csv('Submission.csv', index=False)
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=50)
decision_tree.fit(train_data1,target1)
test_data1=test.drop('pet_id', axis=1)

predictions=decision_tree.predict(test_data1)
submission1=pd.DataFrame({"pet_id":test["pet_id"],"pet_category":predictions})
submission1.to_csv('Submission1.csv', index=False)
output1=pd.read_csv('./Submission.csv')
output2=pd.read_csv('./Submission1.csv')


output1

output2
output1['pet_category']=output2['pet_category']
output1
output1.to_csv('finall.csv', index = False)
