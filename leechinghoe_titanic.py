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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics, svm



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")

data_cleaner=[train,test]
#data cleaning

print('Train columns with null values:\n', train.isnull().sum())

print("-"*10)

print("Test columns with null values:\n",test.isnull().sum())
columns_drop=['Ticket']

for data in data_cleaner:

    data.drop(columns_drop,axis=1,inplace=True)
normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}
#feature engineering

for data in data_cleaner:

    data['Family']=data['SibSp']+data['Parch']+1

    data['Alone']=1

    data['Alone'].loc[data['Family']>1]=0

    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    data.Title = data.Title.map(normalized_titles)
train.Title.value_counts()
for data in data_cleaner:

    data.Age.fillna(data.Age.median(),inplace=True)
for data in data_cleaner:

    data.Cabin.fillna('U',inplace=True)
#imputer fare and embarked

for data in data_cleaner:

    data.Fare.fillna(data.Fare.median(),inplace=True)

    data.Embarked.fillna(data.Embarked.mode()[0],inplace=True)
#Drop Name, SibSp, Parch

drop_col=['Name','SibSp','Parch']

for data in data_cleaner:

    data.drop(drop_col,axis=1,inplace=True)
#extract first letter and fill in unknown for nan

for data in data_cleaner:

    data.fillna('U',inplace=True)

    data.Cabin=data.Cabin.map(lambda x:x[0])
#one hot encode Sex,Cabin,Title,Embarked

OH_cols=['Sex','Cabin','Embarked','Title']

enc=OneHotEncoder(handle_unknown='ignore',sparse=False)

OH_cols_train=pd.DataFrame(enc.fit_transform(train[OH_cols]))

OH_cols_train.columns=enc.get_feature_names(OH_cols)

OH_cols_test=pd.DataFrame(enc.transform(test[OH_cols]))

OH_cols_test.columns=enc.get_feature_names(OH_cols)





OH_cols_train.index=train.index

OH_cols_test.index=test.index
#joining them back and assign new df

train=pd.concat([train,OH_cols_train],axis=1)

test=pd.concat([test,OH_cols_test],axis=1)
df=[train,test]

for ind in df:

    ind.drop(OH_cols,axis=1,inplace=True)
#bin Fare

for ind in df:

    print(ind.info())
def fare_bin(x):

    if x <= 7.91:

        return 1

    elif x > 7.91 and x <= 14.454:

        return 2

    elif  x > 14.454 and x <= 31:

        return 3

    elif x > 31 and x < 513:

        return 4
train.head()
for ind in df:

    ind.Fare=ind.Fare.map(lambda x: fare_bin(x))


def age_bin(x):

    if x <=10:

        return 0

    elif x>10 and x<=20:

        return 1

    elif x>20 and x<=30:

        return 2

    elif x>30 and x<=40:

        return 3

    elif x>40 and x<=50:

        return 4

    elif x>50 and x<=60:

        return 5

    elif x>60 and x<=70:

        return 6

    elif x>70 and x<=80:

        return 7

    elif x>80 and x<=90:

        return 8
for ind in df:

    ind.Age=ind.Age.map(lambda x: age_bin(x))
test.info()
train.columns
len(test.columns)
len(train.columns)
feature_col=['Pclass', 'Age', 'Fare', 'Family', 'Alone',

       'Sex_female', 'Sex_male', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D',

       'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_U', 'Embarked_C',

       'Embarked_Q', 'Embarked_S', 'Title_Master', 'Title_Miss', 'Title_Mr',

       'Title_Mrs', 'Title_Officer', 'Title_Royalty']
target=train['Survived']

features_train=train[feature_col]

X_test=test.drop('PassengerId',axis=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
rf_params=dict(     

    max_depth = [n for n in range(11, 16)],     

    min_samples_split = [n for n in range(6, 13)], 

    min_samples_leaf = [n for n in range(4, 7)],     

    n_estimators = [n for n in range(30, 80, 10)],

)
rf_model=RandomForestClassifier()

rf=GridSearchCV(estimator=rf_model,param_grid=rf_params,cv=5,n_jobs=-1)

rf.fit(features_train,target)
print("Best score: {}".format(rf.best_score_))

print("Optimal params: {}".format(rf.best_estimator_))
gbcgs_params = {

    'loss' : ["deviance","exponential"],

    'n_estimators' : [30,40,50,60],

    'learning_rate': [0.02,0.03,0.04,0.05,0.06],

    'max_depth':  [2,3,4],

    'max_features': [2,3,4],

    "min_samples_split": [2,3,4],

    'min_samples_leaf': [2,3,4]

}
gb_model=GradientBoostingClassifier()

gb=GridSearchCV(estimator=gb_model,param_grid=gbcgs_params,cv=5,n_jobs=-1)

gb.fit(features_train,target)
print("Best score: {}".format(gb.best_score_))

print("Optimal params: {}".format(gb.best_estimator_))
svc_params = {

    'kernel': ['rbf'],

    'C':     [10,20,30,40,50,60,70],

    'gamma': [0.005,0.006,0.007,0.008,0.009,0.01,0.011],

    'probability': [True]

}
svc_model=svm.SVC()

svc=GridSearchCV(svc_model,svc_params,cv=5,n_jobs=-1)

svc.fit(features_train,target)
print("Best score: {}".format(svc.best_score_))

print("Optimal params: {}".format(svc.best_estimator_))
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters

ex_params = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}



gsExtC = GridSearchCV(ExtC,param_grid = ex_params, cv=5, scoring="accuracy", n_jobs= 10, verbose = 1)

gsExtC.fit(features_train,target)

ExtC_best = gsExtC.best_estimator_
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_params = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_params, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(features_train,target)



ada_best = gsadaDTC.best_estimator_
ensemble=VotingClassifier(estimators=[('RF',rf.best_estimator_),('GB',gb.best_estimator_),('SVC',svc.best_estimator_),('extc', ExtC_best),('adac',ada_best)],voting='soft')
ensemble.fit(features_train,target)
pred=ensemble.predict(X_test)
submission=pd.DataFrame({'PassengerId':test.PassengerId,'Survived':pred})
filename = 'my_submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)
