#importing the libraries

import pandas as pd # data processing I\O CSV

import numpy as np #linear algebra



#visualization

import matplotlib.pyplot as plt

import seaborn as sns



#import machine learning models

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix , accuracy_score

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

import warnings

warnings.filterwarnings('ignore')



#import dataset

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')

all_data=[df_train,df_test]
#to show some data 

df_train.head()
#statistical describtion for numerical data

df_train.describe()
#statistical describtion for categorical data

df_train.describe(include='O')
#to show information for data

df_train.info()

print('='*50)

df_test.info()
#in this lesson we can sum number of nulls

total=df_train.isnull().sum().sort_values(ascending=False)

percent=df_train.isnull().sum()/df_train.isnull().count().sort_values(ascending=False)

missing_data=pd.concat([total,percent],axis=1,keys=['total','percent'])

missing_data.head(10)
#we can replace the name with the title i take this idea from 'Manav Sehgal'

for dataset in all_data:

    dataset['Title']=dataset['Name'].str.extract(' ([A-Z a-z]+)\.')

#describe the title

pd.crosstab(dataset['Title'],dataset['Sex'])
list1=['Rev','Dr','Major','Col','Mlle','Lady','Capt','Mme','Jonkheer','Sir','Ms','the Countess','Don','Dona']

for dataset in all_data :

    dataset['Title']=dataset['Title'].replace(list1,'Rare')



df_train[['Title','Survived']].groupby(['Title'],as_index=False).mean().sort_values(by='Survived',ascending=False)
g=sns.FacetGrid(df_train,col='Survived')

g.map(plt.hist,'Title')
for dataset in all_data:

    dataset['Title']=dataset['Title'].map({'Mrs': 1 ,'Miss': 2 ,'Master': 3 ,'Mr': 4 ,'Rare': 5 })

df_train.head(10)
df_test['Fare'].fillna(df_test['Fare'].value_counts().index[0],inplace=True)

df_train['Embarked'].fillna(df_train['Embarked'].value_counts().index[0],inplace=True)

df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
g=sns.FacetGrid(df_train,col='Survived',row='Sex')

g.map(plt.hist,'Embarked')
for dataset in all_data:

    dataset['Embarked']=dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df_train['split Fare']=pd.qcut(df_train['Fare'],4)

df_train[['split Fare','Survived']].groupby(['split Fare'],as_index=False).mean().sort_values(by='split Fare',ascending=True)
for dataset in all_data:

    dataset.loc[(dataset['Fare'] <= 7.896) ,'Fare'] =0

    dataset.loc[(dataset['Fare'] > 7.896) & (dataset['Fare'] <= 14.454) , 'Fare'] =1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.275) , 'Fare'] =2

    dataset.loc[(dataset['Fare'] > 31.275) & (dataset['Fare'] <= 512.329), 'Fare'] =3

    dataset.loc[(dataset['Fare'] > 512.329) , 'Fare'] =4
df_train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
g=sns.FacetGrid(df_train,col='Survived',row='Sex')

g.map(plt.hist,'Age',bins=20)
for dataset in all_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} )
mean_ages=np.zeros((2,3))



for datasets in all_data:

    for i in range(0,2):

        for j in range(0,3):

            Class=datasets[(datasets['Pclass']==j+1) & (datasets['Sex']==i) ]['Age']

            mean_ages[i,j]=Class.mean()

    for i in range(0,2):

        for j in range(0,3):

            datasets.loc[(datasets['Pclass']==j+1) & (datasets['Sex']==i) & (datasets['Age'].isnull()) , 'Age']=mean_ages[i,j]

    datasets['Age'] = datasets['Age']

    

df_train['Age'].isnull().sum()

            

        

            
g=sns.FacetGrid(df_train,col='Pclass',row='Survived')

g.map(plt.hist,'Age',bins=20)
df_train['split Age']=pd.cut(df_train['Age'],5)

df_train[['split Age','Survived']].groupby(['split Age'],as_index=False).mean().sort_values(by='split Age',ascending=True)
for dataset in all_data:    

    dataset.loc[ (dataset['Age']) <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

df_train.head()
df_train = df_train.drop(['PassengerId','Name','Ticket', 'Cabin','split Fare','split Age'], axis=1)

df_test = df_test.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
#we will show correlation matrix

corrmat=df_train.corr()

f,ax=plt.subplots(figsize=(12,9))

sns.heatmap(corrmat,cmap='RdYlGn',annot=True)
corrmat['Survived'].sort_values(ascending=False)
X=df_train.drop('Survived',axis=1)

y=df_train['Survived']
X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=150,test_size=0.2)

#applying Support Vector Machines

svc = SVC()

kernel=[ 'linear','poly' ,'rbf', 'sigmoid']

param=dict(kernel=kernel)

SVCModel=GridSearchCV(estimator=svc,param_grid=param,cv=10,n_jobs=-1)

SVCModel.fit(X_train,y_train)

y_pred=SVCModel.predict(X_val)

print('best parameters : ',SVCModel.best_params_)

y_pred=SVCModel.predict(X_val)

print('SVCModel score train : ',SVCModel.score(X_train,y_train))

print('SVCModel score test : ',accuracy_score(y_val,y_pred)) 

print('confusion matrix : \n',confusion_matrix(y_val,y_pred))



print('='*50)



#applying k Nerest Neighbors

knn = KNeighborsClassifier(n_neighbors = 5)



n_neighbors=[3,5,7,9]

algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']

weights=['uniform','distance']

param=dict(n_neighbors=n_neighbors,algorithm=algorithm,weights=weights)

KNNModel=GridSearchCV(estimator=knn,param_grid=param,cv=10,n_jobs=-1)

KNNModel.fit(X_train,y_train)

print('best parameters : ',KNNModel.best_params_)

y_pred=KNNModel.predict(X_val)

print('KNNModel score train : ',KNNModel.score(X_train,y_train))

print('KNNModel score test : ',accuracy_score(y_val,y_pred)) 

print('confusion matrix : \n',confusion_matrix(y_val,y_pred))



print('='*50)



#applying Random Forest

random_forest = RandomForestClassifier(max_depth=10,random_state=100)

random_forest.fit(X_train,y_train)

y_pred=random_forest.predict(X_val)

print('random_forest score train : ',random_forest.score(X_train,y_train))

print('random_forest score test : ',accuracy_score(y_val,y_pred)) 

print('confusion matrix : \n',confusion_matrix(y_val,y_pred))



print('='*50)





#applying GBoost Classifier

xgb=XGBClassifier(max_depth=8).fit(X_train,y_train)

y_pred=xgb.predict(X_val)

print('xgb score train : ',xgb.score(X_train,y_train))

print('xgb score test : ',accuracy_score(y_val,y_pred)) 

print('confusion matrix : \n',confusion_matrix(y_val,y_pred))



X=df_train.drop('Survived',axis=1)

y=df_train['Survived']

X_test=df_test
#applying Support Vector Machines

svc = SVC()

kernel=[ 'linear','poly' ,'rbf', 'sigmoid']

param=dict(kernel=kernel)

SVCModel=GridSearchCV(estimator=svc,param_grid=param,cv=10,n_jobs=-1)

SVCModel.fit(X,y)

print('best parameters : ',SVCModel.best_params_)

y_pred=SVCModel.predict(X_test)

print('SVCModel score train : ',SVCModel.score(X,y))





print('='*50)



#applying k Nerest Neighbors

knn = KNeighborsClassifier(n_neighbors = 5)



n_neighbors=[3,5,7,9]

algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']

weights=['uniform','distance']

param=dict(n_neighbors=n_neighbors,algorithm=algorithm,weights=weights)

KNNModel=GridSearchCV(estimator=knn,param_grid=param,cv=10,n_jobs=-1)

KNNModel.fit(X,y)

print('best parameters : ',KNNModel.best_params_)

y_pred=KNNModel.predict(X_test)

print('KNNModel score train : ',KNNModel.score(X,y))



print('='*50)



#applying Random Forest

random_forest = RandomForestClassifier(max_depth=10,random_state=100)

random_forest.fit(X,y)

y_pred=random_forest.predict(X_test)

print('random_forest score train : ',random_forest.score(X,y))





print('='*50)





#applying GBoost Classifier

xgb=XGBClassifier(max_depth=8).fit(X_train,y_train)

y_pred=xgb.predict(X_test)

print('xgb score train : ',xgb.score(X,y))




