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

#importing required library

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import  LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.datasets import make_classification
#reading the  train data

df_train=pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.head()
#read the test data

df_test=pd.read_csv('/kaggle/input/titanic/test.csv')

#checking the number of nan values

df_train.isnull().sum()
#understanding outlairs

df_train.plot(kind ='box', subplots = True, layout =(3, 3),  

                       sharex = False, sharey = False,figsize=(20,20)) 
#understanding no of survived persons

sns.countplot(x='Survived', data=df_train);

#plotting between sex and survived

sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train)
#understanding the no; male and female survived

df_train.groupby(['Sex']).Survived.sum()
#plotting between pclass and survived

sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);
#plotting between embarked and survived

sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);
#understanding the correlation

plt.figure(figsize=(12, 8))

sns.heatmap(df_train.corr(), cmap=plt.cm.RdBu, annot=True)
#checking the unique values in the sex column

np.unique(df_train.Sex)

#repalce the string value of sex with a numerical value

a= [1,2]

b=['female', 'male']

df_train=df_train.replace(b,a)

df_train.head()
#replace the catogorical value with numerical

df_test=df_test.replace(b,a)

df_train = df_train[pd.notnull(df_train['Embarked'])]

#undersanding unique values

np.unique(df_train.Embarked)

#applying label encoding

label = LabelEncoder()



df_train['Embarked']= label.fit_transform(df_train['Embarked']) 



    

#replace nan value with mean of the feature

df_train['Age']=df_train['Age'].fillna(df_train['Age'].mean()) 

#understanding nan value

df_train.isnull().sum()

x_train=df_train[['Pclass','Sex','Age','SibSp','Parch',"Embarked"]]

y_train=df_train['Survived']

df_test.isnull().sum()
#replace nan value with mean of the feature

df_test['Age']=df_test['Age'].fillna(df_test['Age'].mean()) 

#replace nan value with mean of the feature

df_train['Fare']=df_train['Fare'].fillna(df_train['Fare'].mean()) 

label = LabelEncoder()



df_test['Embarked']= label.fit_transform(df_test['Embarked']) 
x_test=df_test[['Pclass','Sex','Age','SibSp','Parch','Embarked']]

scaler = MinMaxScaler()

scaler.fit(x_train)
gbrt = GradientBoostingClassifier()

learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

param_grid = dict(learning_rate=learning_rate)

gbrt1 = GridSearchCV(gbrt, param_grid)

gbrt1.fit(x_train,y_train)

gbrt2=GradientBoostingClassifier(learning_rate= 0.1)

scores = cross_val_score(gbrt2, x_train, y_train, cv=5)

print(scores.mean() )
ada=AdaBoostClassifier()

learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

param_grid = dict(learning_rate=learning_rate)

ada1 = GridSearchCV(ada, param_grid)

ada1.fit(x_train, y_train)

ada1.best_params_

ada2=AdaBoostClassifier(learning_rate= 0.1)

scores0 = cross_val_score(ada2, x_train, y_train, cv=5)

print(scores0.mean() ) 
lg=LogisticRegression()

penalty = ['l1', 'l2']



# Create regularization hyperparameter space

C = np.logspace(0, 4, 10)



# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)

lg1= GridSearchCV(lg, hyperparameters, cv=5, verbose=0)



lg1.fit(x_train, y_train)  

lg1.best_params_
lg2=LogisticRegression(C=1,penalty='l2')

scores1 = cross_val_score(lg2, x_train, y_train, cv=5)

print(scores1.mean() )             
param_gridc = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}  

  

grid = GridSearchCV(SVC(), param_gridc, refit = True, verbose = 3) 



grid.fit(x_train, y_train)  

grid.best_params_
grid1=GridSearchCV(SVC(C=1000,gamma=.001,kernel='rbf'), param_gridc, refit = True, verbose = 3)

grid1.fit(x_train, y_train)

score2=cross_val_score(grid1, x_train, y_train, cv=5)

print(score2.mean())
rf=RandomForestClassifier(random_state=42)

param_grid = { 

    'n_estimators': [200, 500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}

CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)

CV_rf.fit(x_train, y_train)

CV_rf.best_params_

rf1=RandomForestClassifier(random_state=42, max_features= 'auto',n_estimators= 200,max_depth= 4,criterion= 'entropy')

rf1.fit(x_train, y_train)

score3=cross_val_score(rf1, x_train, y_train, cv=5)

print(score3.mean())
#here random forest is more accurate

y_pred=rf1.predict(x_test)
gender_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived':y_pred})



gender_submission.to_csv('gender_submission.csv')
gender_submission.head(5)