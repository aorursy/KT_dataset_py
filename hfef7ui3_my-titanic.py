import pandas as pd

import numpy as np

import tensorflow as tf



data_train=pd.read_csv("../input/train.csv")

data_test=pd.read_csv("../input/test.csv")

print(data_train)
import matplotlib as plt
data_train=data_train.drop(['Name','Ticket'],axis=1)

data_test=data_test.drop(['Name','Ticket'],axis=1)

#print(data_train)

from sklearn.ensemble import RandomForestRegressor

# import sklearn.ensemble

def set_missing_ages(df):

    age_df=df[['Age','Fare','Parch','SibSp','Pclass']]

    

    known_age=age_df[age_df.Age.notnull()].as_matrix()

    unknown_age=age_df[age_df.Age.isnull()].as_matrix()

    

    print(known_age)

    y=known_age[:,0]

    x=known_age[:,1:]

    # print(x)

    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=1)

    rfr.fit(x,y)



    predictedAges=rfr.predict(unknown_age[:,1::])

    df.loc[(df.Age.isnull()),'Age']=predictedAges

    return df,rfr



def set_Cabin_type(df):

    df.loc[(df.Cabin.notnull()),'Cabin']='Yes'

    df.loc[(df.Cabin.isnull()),'Cabin']='No'

    return df



data_train,rfr=set_missing_ages(data_train)

data_train=set_Cabin_type(data_train)
dummies_Cabin=pd.get_dummies(data_train['Cabin'],prefix='Cabin')

dummies_Embarked=pd.get_dummies(data_train['Embarked'],prefix='Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')



df=pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)

df.drop(['Pclass','Sex','Cabin','Embarked'],axis=1,inplace=True)

print(df)
import sklearn.preprocessing as preprocessing

scaler=preprocessing.StandardScaler()

age_scale_param=scaler.fit(df['Age'].values.reshape(-1,1))

df['Age_scaled']=scaler.fit_transform(df['Age'].values.reshape(-1,1),age_scale_param)

fare_scale_param=scaler.fit(df['Fare'].values.reshape(-1,1))

df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)

print(df)
from sklearn import linear_model

train_df=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

train_np=train_df.as_matrix()



y=train_np[:,0]

x=train_np[:,1:]

clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)

clf.fit(x,y)
data_test.loc[(data_test.Fare.isnull()),'Fare']=0

tmp_df=data_test[['Age','Fare','Parch','SibSp','Pclass']]

null_age=tmp_df[(data_test.Age.isnull())].as_matrix()

x=null_age[:,1:]

predictedAges=rfr.predict(x)

data_test.loc[(data_test.Age.isnull()),'Age']=predictedAges

data_test=set_Cabin_type(data_test)

dummies_Cabin=pd.get_dummies(data_test['Cabin'],prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test=pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)

df_test=df_test.drop(['Pclass','Sex','Cabin','Embarked'],axis=1)

df_test['Age_scaled']=scaler.fit_transform(df_test['Age'].values.reshape(-1,1),age_scale_param)

df_test['Fare_scaled']=scaler.fit_transform(df_test['Fare'].values.reshape(-1,1),fare_scale_param)

print(df_test)
test=df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions=clf.predict(test)

result=pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})

result.to_csv('./MyTitanicTest0.csv',index=False)