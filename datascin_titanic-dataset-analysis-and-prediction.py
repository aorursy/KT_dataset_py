# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import os
train_df = pd.read_csv("../input/train.csv", index_col='PassengerId')
test_df = pd.read_csv("../input/test.csv", index_col='PassengerId')
train_df.info()
test_df.info()
test_df['Survived']= -888 #adding survived with a default value
df = pd.concat((train_df, test_df) , axis=0)
df.info()
df.head()
df[df.Embarked.isnull()]
df.Embarked.value_counts()
pd.crosstab(df[df.Survived !=-888].Survived,df[df.Survived != -888].Embarked)
df.groupby(['Pclass', 'Embarked']).Fare.median()
df.Embarked.fillna('C', inplace=True)
df[df.Embarked.isnull()]
df.info()
df[df.Fare.isnull()]
median_fare=df.loc[(df.Pclass ==3) & (df.Embarked == 'S'),'Fare'].median()
print(median_fare)
df.Fare.fillna(median_fare, inplace=True)
pd.options.display.max_rows =15
df[df.Age.isnull()]
df.Age.plot(kind='hist',bins=20,color='c')
df.groupby('Sex').Age.median()
df[df.Age.notnull()].boxplot('Age','Sex')
df[df.Age.notnull()].boxplot('Age','Pclass')
def GetTitle(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]
df['Title']= df.Name.map(lambda x :GetTitle(x))
df[df.Age.notnull()].boxplot('Age','Title')
title_age_median=df.groupby('Title').Age.transform('median')
df.Age.fillna(title_age_median, inplace=True)
df.loc[df.Age>70]
df.loc[df.Fare == df.Fare.max()]
LogFare=np.log(df.Fare+1.0)
LogFare.plot(kind='hist')
pd.qcut(df.Fare,4)
pd.qcut(df.Fare,4,labels=['very low','low','high','very high'])
pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high']).value_counts().plot(kind='bar', color='c', rot=0);
df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high'])
df['AgeState']=np.where(df['Age']>=18,'Adult','Child')
df['AgeState'].value_counts()
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].AgeState)
df['FamilySize']=df.Parch+df.SibSp+1
df['FamilySize'].plot(kind='hist')
df.loc[df.FamilySize==df.FamilySize.max(),['Name','Survived','FamilySize','Ticket']]
pd.crosstab(df[df.Survived !=-888].Survived,df[df.Survived!=-888].FamilySize)
df['IsMother']=np.where(((df.Sex=='female')&(df.Parch>0)&(df.Age>18)&(df.Title!='Miss')),1,0)
pd.crosstab(df[df.Survived !=-888].Survived,df[df.Survived!=-888].IsMother)
df.Cabin
df.Cabin.unique()
df.loc[df.Cabin=='T']
df.loc[df.Cabin=='T','Cabin']=np.NaN
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')
df['Deck']= df['Cabin'].map(lambda x: get_deck(x))
df.Deck.value_counts()
pd.crosstab(df[df.Survived !=-888].Survived, df[df.Survived !=-888].Deck)
df['IsMale']=np.where(df.Sex=='Male',1,0)
df=pd.get_dummies(df,columns=['Deck','Pclass','Title','Fare_Bin','Embarked','AgeState'])
print(df.info())
# drop and reorder columns
df.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1,inplace=True)
# reorder columns
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived'] + columns
df = df[columns]
# check info again
df.info()
# train data
train_df= df[df.Survived != -888]
# test data
columns = [column for column in df.columns if column != 'Survived']
test_df= df[df.Survived == -888][columns]
train_df.info()
test_df.info()
X = train_df.loc[:,'Age':].as_matrix().astype('float')
y = train_df['Survived'].ravel()
print(X.shape,Y.shape)
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# average survival in train and test
print('mean survival in train : {0:.3f}'.format(np.mean(y_train)))
print('mean survival in test : {0:.3f}'.format(np.mean(y_test)))
import sklearn
from sklearn.dummy import DummyClassifier
# create model
model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)
# train model
model_dummy.fit(X_train, y_train)
print('score for baseline model : {0:.2f}'.format(model_dummy.score(X_test, y_test)))
# peformance metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
print('Accuracy for baseline model {:.2f}'.format(accuracy_score(Y_test,model_dummy.predict(X_test))))
print('Confusion Matrix baseline model\n {}'.format(confusion_matrix(Y_test,model_dummy.predict(X_test))))
print('precision for basleline model :{:.2f}'.format(precision_score(Y_test,model_dummy.predict(X_test))))
# import function
from sklearn.linear_model import LogisticRegression
# create model
model_lr_1= LogisticRegression(random_state=0)
# train model
model_lr_1.fit(X_train, y_train)
#evaluate model
print('score for logistic regression - version 1 : {0:.2f}'.format(model_lr_1.score(X_test,Y_test)))
# performance metrics
# accuracy
print('accuracy for logistic regression - version 1 : {0:.2f}'.format(accuracy_score(Y_test, model_lr_1.predict(X_test))))
# confusion matrix
print('confusion matrix for logistic regression - version 1: \n {0}'.format(confusion_matrix(Y_test, model_lr_1.predict(X_test))))
# precision 
print('precision for logistic regression - version 1 : {0:.2f}'.format(precision_score(Y_test, model_lr_1.predict(X_test))))
# precision 
print('recall for logistic regression - version 1 : {0:.2f}'.format(recall_score(Y_test, model_lr_1.predict(X_test))))
# model coefficients
model_lr_1.coef_
# base model 
model_lr = LogisticRegression(random_state=0)
from sklearn.model_selection import GridSearchCV
parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}
clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)
clf.fit(X_train, y_train)
clf.best_params_
print('best score : {0:.2f}'.format(clf.best_score_))
# evaluate model
print('score for logistic regression - version 2 : {0:.2f}'.format(clf.score(X_test, y_test)))
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# feature normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled[:,0].min(),X_train_scaled[:,0].max()
# normalize test data
X_test_scaled = scaler.transform(X_test)
# feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# base model 
model_lr = LogisticRegression(random_state=0)
parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}
clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)
clf.fit(X_train_scaled, y_train)
clf.best_score_
# evaluate model
print('score for logistic regression - version 2 : {0:.2f}'.format(clf.score(X_test_scaled, y_test)))
