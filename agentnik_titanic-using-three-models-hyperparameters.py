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
# load dataset

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

submission=pd.read_csv("../input/titanic/gender_submission.csv")
train.head()
len(train['Ticket'].unique().tolist())

#Dropping Name and Ticket column

train.drop(['Name','Ticket'], axis=1,inplace=True)

test.drop(['Name','Ticket'], axis=1,inplace=True)
train.describe
train.info()
train.shape
test.shape
#checking null values

train.isnull().sum()
test.isnull().sum()
#Visualization for null values

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno
msno.bar(train,figsize=(8,4),color="Red")

plt.show()
msno.bar(test,figsize=(8,4),color="Maroon")

plt.show()
# replacing nulls in age with median which ignores outliers

train['Age'].fillna(train['Age'].median(),inplace=True)

test['Age'].fillna(train['Age'].median(),inplace=True)



train['Age']
#Replacing Null with Unknown category for Cabin and Embarked

train['Cabin'].fillna('Unknown',inplace=True)

train['Embarked'].fillna('Unknown',inplace=True)

test['Cabin'].fillna('Unknown',inplace=True)



#Replacing Null with Median value of Fare from train data

test['Fare'].fillna(train['Fare'].median(),inplace=True)
msno.bar(train,figsize=(8,4),color="blue")

plt.show()
msno.bar(test,figsize=(8,4),color="darkblue")

plt.show()
#Checking response variable

train['Survived'].value_counts(normalize=True)*100
train.shape
test.shape
#adding Survived column in test

test['Survived'] = np.nan

#combining train and test 

train['data'] = 'train'

test['data'] = 'test'

test=test[train.columns] 

t_all=pd.concat([train,test],axis=0)
t_all.info
#Creating Dummies

for col in ['Sex','Cabin','Embarked']:

    temp=pd.get_dummies(t_all[col],prefix=col,drop_first=True)

    t_all=pd.concat([temp,t_all],1)

    t_all.drop([col],1,inplace=True)
t_all.dtypes
t_all.shape
# Separate the train and test data

train=t_all[t_all['data']=='train']

del train['data'] 



test=t_all[t_all['data']=='test'] 

test.drop(['Survived','data'],axis=1,inplace=True)
train.shape
test.shape
# X and Y to train model

X = train.drop('Survived',axis=1)

Y = train['Survived']
#Model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



#Model Select

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split



#Metrics

from sklearn.metrics import make_scorer, accuracy_score,precision_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

#Divivding Data further for Validation

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=900)


knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)



Y_pred1 = knn.predict(X_test) 

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
dtree = DecisionTreeClassifier() 

dtree.fit(X_train , Y_train)  



Y_pred2 = dtree.predict(X_test) 

acc_dtree = round(dtree.score(X_train , Y_train) * 100, 2)
rforest = RandomForestClassifier(n_estimators=200)

rforest.fit(X_train , Y_train)



Y_pred3 = rforest.predict(X_test)

rforest.score(X_train , Y_train)

acc_rforest = round(rforest.score(X_train , Y_train) * 100, 2)
results = pd.DataFrame({

    'Model': [ 'KNN', 'Decision Tree', 'Random Forest' ],

    'Score': [ acc_knn , acc_dtree , acc_rforest ]})

              

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(5)
#Decision Tree with Hyperparameter Tuning

clf = DecisionTreeClassifier()
params={'class_weight':[None,'balanced'], 

        'criterion':['entropy','gini'],

        'max_depth':[None,5,10,15,20,30,50,70],

        'min_samples_leaf':[1,2,5,10,15,20], 

        'min_samples_split':[2,5,10,15,20]

       }
2*2*8*6*5
#randomized search with 10% of total

random_search=RandomizedSearchCV(clf,cv=5,

                                 param_distributions=params,

                                 scoring='roc_auc',

                                 n_iter=96,

                                 n_jobs=-1,

                                 verbose=5

                                    )
random_search.fit(X,Y)
random_search.best_estimator_
# Scores for Model



def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.5f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")



report(random_search.cv_results_,5)
#Fitting model with Hyperparametes

dtree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, min_samples_split=10)
dtree.fit(X,Y)
#Prediction on test data

submission1 = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": dtree.predict(test)

})
submission['Survived'].value_counts()
submission1['Survived'].value_counts()
submission1['Survived'].value_counts(normalize=True)*100
submission1.astype(int)
#submitting as csv for competing

submission1.astype(int).to_csv('submission.csv', index=False)