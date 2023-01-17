# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#linear algebra

import numpy as np



#dataframe

import pandas as pd



#data visualization

import matplotlib.pyplot as plt

import seaborn as sns



#regex

import re

import sklearn

#machine learning

from sklearn.preprocessing import LabelEncoder,MinMaxScaler

from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report,f1_score,accuracy_score

from xgboost import XGBClassifier



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



from vecstack import stacking



#neural network

import tensorflow as tf

from tensorflow import keras



import missingno



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd

from scipy import stats

import copy 

import warnings

warnings.filterwarnings('ignore')



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine-learning

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from sklearn.feature_selection import RFE, RFECV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier

#from xgboost import XGBClassifier

#from imblearn.over_sampling import SMOTE

#from sklearn.ensemble import BaggingClassifier
train = pd.read_csv('../input/prsits/train.csv')

test = pd.read_csv('../input/prsits/test.csv')
missingno.matrix(train, figsize = (10,5))
missingno.matrix(test, figsize = (10,5))
train.head()
train.LIMIT_BAL.describe()
test.head()
train.head()
# create bins for age

def age_group_fun(age):

    a = ''

    if age <= 0:

        a = 0

    else:

        a = 1

    return a

        

## Applying "age_group_fun" function to the "Age" column.

train['PAY_0'] = train['PAY_0'].map(age_group_fun)

test['PAY_0'] = test['PAY_0'].map(age_group_fun)

train['PAY_2'] = train['PAY_2'].map(age_group_fun)

test['PAY_2'] = test['PAY_2'].map(age_group_fun)

train['PAY_3'] = train['PAY_3'].map(age_group_fun)

test['PAY_3'] = test['PAY_3'].map(age_group_fun)

train['PAY_4'] = train['PAY_4'].map(age_group_fun)

test['PAY_4'] = test['PAY_4'].map(age_group_fun)

train['PAY_5'] = train['PAY_5'].map(age_group_fun)

test['PAY_5'] = test['PAY_5'].map(age_group_fun)

train['PAY_6'] = train['PAY_6'].map(age_group_fun)

test['PAY_6'] = test['PAY_6'].map(age_group_fun)
train.rename({'default.payment.next.month' : 'skip'}, axis=1, inplace=True)
def sum_frame_by_column(frame, new_col_name, list_of_cols_to_sum):

    frame[new_col_name] = frame[list_of_cols_to_sum].astype(float).sum(axis=1)

    return(frame)
sum_frame_by_column(train, 'sum_pay', ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5', 'PAY_AMT6'])

sum_frame_by_column(train, 'delay', ['PAY_0', 'PAY_2', 'PAY_3','PAY_4', 'PAY_5','PAY_6'])

sum_frame_by_column(train, 'sum_bill', ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5', 'BILL_AMT6'])
sum_frame_by_column(test, 'sum_pay', ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5', 'PAY_AMT6'])

sum_frame_by_column(test, 'sumpay', ['PAY_0', 'PAY_2', 'PAY_3','PAY_4', 'PAY_5','PAY_6'])

sum_frame_by_column(test, 'sum_bill', ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5', 'BILL_AMT6'])
def mean_frame_by_column(frame, new_col_name, list_of_cols_to_mean):

    frame[new_col_name] = frame[list_of_cols_to_mean].astype(float).mean(axis=1)

    return(frame)
#mean_frame_by_column(test, 'skip_prob', ['PAY_0', 'PAY_2', 'PAY_3','PAY_4', 'PAY_5','PAY_6'])

#mean_frame_by_column(train, 'skip_prob', ['PAY_0', 'PAY_2', 'PAY_3','PAY_4', 'PAY_5','PAY_6'])
train.head()
train.groupby(["ID"]).mean()["skip"]
train.rename({'default.payment.next.month' : 'skip'}, axis=1, inplace=True)
train = train.drop(["BILL_AMT6","BILL_AMT5","BILL_AMT4","BILL_AMT3","BILL_AMT2","BILL_AMT1","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","ID"],axis=1)
#train = train.drop(["BILL_AMT6","BILL_AMT5","BILL_AMT4","BILL_AMT3","BILL_AMT2","BILL_AMT1","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","ID"],axis=1)
test.head(10)
train.rename({'default.payment.next.month' : 'skip'}, axis=1, inplace=True)
train
A=test["ID"]


test = test.drop(["BILL_AMT6","BILL_AMT5","BILL_AMT4","BILL_AMT3","BILL_AMT2","BILL_AMT1","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","ID"],axis=1)





#test = test.drop(["BILL_AMT6","BILL_AMT5","BILL_AMT4","BILL_AMT3","BILL_AMT2","BILL_AMT1","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"],axis=1)
test
train
test.head()
train.sample(100)
pd.DataFrame(abs(train.corr()['skip']).sort_values(ascending = False))
# Pointplot Pclass type

ax2 = plt.subplot(2,1,2)

sns.pointplot(x='SEX', y='skip', data=train)

ax2.set_xlabel('SEX')

ax2.set_ylabel('Percent Skip')

ax2.set_title('Percentage Skip by SEX')
ax2 = plt.subplot(2,1,2)

sns.pointplot(x='MARRIAGE', y='skip', data=train)
ax2 = plt.subplot(2,1,2)

sns.pointplot(x='EDUCATION', y='skip', data=train)
fig = plt.figure(figsize=(12,6),)

ax=sns.kdeplot(train.loc[(train['skip'] == 0),'AGE'] , color='gray',shade=True,label='not skip')

ax=sns.kdeplot(train.loc[(train['skip'] == 1),'AGE'] , color='g',shade=True, label='skip')

plt.title('Age Distribution - Skip V.S. Non Skip', fontsize = 25, pad = 40)

plt.xlabel("AGE", fontsize = 15, labelpad = 20)

plt.ylabel('Frequency', fontsize = 15, labelpad= 20);
fig = plt.figure(figsize=(12,6),)

ax=sns.kdeplot(train.loc[(train['skip'] == 0),'LIMIT_BAL'] , color='gray',shade=True,label='not skip')

ax=sns.kdeplot(train.loc[(train['skip'] == 1),'LIMIT_BAL'] , color='g',shade=True, label='skip')

plt.title('LIMIT_BAL Distribution - Skip V.S. Non Skip', fontsize = 25, pad = 40)

plt.xlabel("LIMIT_BAL", fontsize = 15, labelpad = 20)

plt.ylabel('Frequency', fontsize = 15, labelpad= 20);
train
le = LabelEncoder()

train['SEX'] = le.fit_transform(train['SEX'])

le = LabelEncoder()

train['EDUCATION'] = le.fit_transform(train['EDUCATION'])

le = LabelEncoder()

train['MARRIAGE'] = le.fit_transform(train['MARRIAGE'])
le = LabelEncoder()

test['SEX'] = le.fit_transform(test['SEX'])

le = LabelEncoder()

test['EDUCATION'] = le.fit_transform(test['EDUCATION'])

le = LabelEncoder()

test['MARRIAGE'] = le.fit_transform(test['MARRIAGE'])
train
pd.DataFrame(abs(train.corr()['skip']).sort_values(ascending = False))
train.head()

test
test
train
X = train.drop(["skip"],axis=1)

y = train["skip"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .25, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



## transforming "train_x"

X_train = sc.fit_transform(X_train)

## transforming "test_x"

X_test = sc.transform(X_test)

# import LogisticRegression model in python. 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error, accuracy_score



## call on the model object

logreg = LogisticRegression(solver='liblinear',

                            penalty= 'l1',

                                

                            )



## fit the model with "train_x" and "train_y"

logreg.fit(X_train,y_train)



## Once the model is trained we want to find out how well the model is performing, so we test the model. 

## we use "test_x" portion of the data(this data was not used to fit the model) to predict model outcome. 

y_pred = logreg.predict(X_test)



## Once predicted we save that outcome in "y_pred" variable.

## Then we compare the predicted value( "y_pred") and actual value("test_y") to see how well our model is performing. 



print ("So, Our accuracy Score is: {}".format(round(accuracy_score(y_pred, y_test),4)))
## Using StratifiedShuffleSplit

## We can use KFold, StratifiedShuffleSplit, StratiriedKFold or ShuffleSplit, They are all close cousins. look at sklearn userguide for more info.   

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

## Using standard scale for the whole dataset.



## saving the feature names for decision tree display

column_names = X.columns



X = sc.fit_transform(X)

accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X,y, cv  = cv)

print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))
from sklearn.model_selection import GridSearchCV, StratifiedKFold

## C_vals is the alpla value of lasso and ridge regression(as alpha increases the model complexity decreases,)

## remember effective alpha scores are 0<alpha<infinity 

C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]

## Choosing penalties(Lasso(l1) or Ridge(l2))

penalties = ['l1','l2']

## Choose a cross validation strategy. 

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)



## setting param for param_grid in GridSearchCV. 

param = {'penalty': penalties, 'C': C_vals}



logreg = LogisticRegression(solver='liblinear')

## Calling on GridSearchCV object. 

grid = GridSearchCV(estimator=LogisticRegression(), 

                           param_grid = param,

                           scoring = 'accuracy',

                            n_jobs =-1,

                           cv = cv

                          )

## Fitting the model

grid.fit(X, y)
logreg_grid = grid.best_estimator_

logreg_grid.score(X,y)
X_train
y_train
y_test
y_pred
submission = pd.DataFrame({

        "id": A,

        "skip": logreg_grid.predict(X_test)

    })

submission.skip = submission.skip



submission.to_csv("fixkk.csv", index=False)
submission.head(10)
test.head(116)