import numpy as np 

import pandas as pd

import sklearn as skl

import warnings

warnings.filterwarnings('ignore')
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.info()
data_train.head()
data_list=[data_train,data_test]

data_combine = pd.concat(data_list, axis=0, join='outer',keys=['data_train','data_test'],ignore_index=False,levels=None,names=None,

                         verify_integrity=False)
data_train.head()
list_columns_type = list(data_combine.columns)

column_int_nan   =[]

column_float_nan =[]

column_object_nan=[]



for i in list_columns_type:

    if data_combine[i].dtypes==int and data_combine[i].isnull().sum()>0:

            column_int_nan.append(i)

    elif data_combine[i].dtypes==float and data_combine[i].isnull().sum()>0:

        column_float_nan.append(i)

    elif data_combine[i].dtypes==object and data_combine[i].isnull().sum()>0:

        column_object_nan.append(i)



column_nan_list=[column_int_nan,column_float_nan,column_object_nan]

for column in column_nan_list:

    print(data_combine[column].isnull().sum())

    print('-'*30)

print(data_combine.shape)



list_Age_was_missing = list(data_combine['Age'].isnull())

list_Cabin_was_missing = list(data_combine['Cabin'].isnull())



list_missing=[list_Age_was_missing,list_Cabin_was_missing]

column_missing=['Age_was_missing','Cabin_was_missing']



for lists,columns in zip(list_missing,column_missing):

    for i in range(len(lists)):

        if lists[i]==False:

            lists[i]=1

        else:

            lists[i]=0

    data_combine[columns]=lists
data_combine['FamilySize']=data_combine['SibSp'] + data_combine['Parch']+1

data_combine['FamilySize'].unique()
data_combine['IsAlone']=1

data_combine['IsAlone'].loc[data_combine['FamilySize'] > 1] = 0
data_combine['Title'] = data_combine['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

data_combine['Title'].unique()
data_combine['CabinHead']=data_combine.Cabin.str[0]

data_combine['CabinHead'].unique()
data_combine=data_combine.join(pd.get_dummies(data_combine['Title'],dtype=int))

data_combine=data_combine.join(pd.get_dummies(data_combine['Sex'],dtype=int))

data_combine=data_combine.join(pd.get_dummies(data_combine['Embarked'],dtype=int))



data_combine.rename(index=str, columns={"S": "Embarked_S", "Q": "Embarked_Q","C":"Embarked_C"},inplace=True)

data_combine=data_combine.join(pd.get_dummies(data_combine['CabinHead'],dtype=int))
data_combine=data_combine.drop(['Title'],axis=1)

data_combine=data_combine.drop(['Name'],axis=1)

data_combine=data_combine.drop(['Sex'],axis=1)

data_combine=data_combine.drop(['Embarked'],axis=1)

data_combine=data_combine.drop(['CabinHead'],axis=1)

data_combine=data_combine.drop(['Cabin'],axis=1)

data_combine=data_combine.drop(['Ticket'],axis=1)
data_combine['Fare'].fillna(data_combine['Fare'].mean(),inplace=True)
data_combine['Fare']=round(data_combine['Fare']).astype(int)
from sklearn.model_selection import train_test_split



y = data_combine[data_combine['Age'].notnull()]['Age']

X = data_combine[data_combine['Age'].notnull()].drop(['Age'],axis=1).drop(['Survived'],axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=33)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error



RFR=RandomForestRegressor()

parameters={'random_state':list(range(10))}



%timeit

GSC= GridSearchCV(RFR,parameters,refit=True,cv=3)

GSC.fit(X_train,y_train)



prediction_test = GSC.predict(X_test)



score = mean_absolute_error(y_test, prediction_test)

prediction = GSC.predict(data_combine[data_combine['Age'].isnull()].drop(['Age'],axis=1).drop(['Survived'],axis=1))

print('MAE:\n',score)

print('best_params_:\n',GSC.best_params_)

print('best_score_:\n',GSC.best_score_)

print('score(X_test,y_test)\n:',GSC.score(X_test,y_test))

print('prediction:\n',prediction)
data_combine['Age'].fillna('NaN',inplace=True)

j=0

for i in range(len(data_combine['Age'])):

    if data_combine['Age'][i]=='NaN':

        data_combine['Age'][i]= prediction[j]

        j=j+1

    data_combine['Age'][i]=round(data_combine['Age'][i])
data_combine['Age']=data_combine['Age'].astype(int)
data_train=data_combine.head(891)

data_test=data_combine.tail(418)



data_test=data_test.drop(['Survived'],axis=1)

data_train['Survived']=data_train['Survived'].astype(int)



data_train.to_csv('train_data.csv')

data_test.to_csv('test_data.csv')
data_train.info()
data_train.head()