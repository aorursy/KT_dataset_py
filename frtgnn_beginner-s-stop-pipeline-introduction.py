import numpy as np 

import pandas as pd



from sklearn.pipeline      import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute        import SimpleImputer

from sklearn.linear_model  import LogisticRegression
df_train = pd.read_csv('../input/titanic/train.csv')

df_test  = pd.read_csv('../input/titanic/test.csv')

df_sample= pd.read_csv('../input/titanic/gender_submission.csv')
df_train.info()
df_train.head()
df_train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

df_test.drop( ['Name','Ticket','Cabin'],axis=1,inplace=True)



# We could create new features from these 3 but I aim to keep it simple and minimal
sex    = pd.get_dummies(df_train['Sex'],drop_first=True)

embark = pd.get_dummies(df_train['Embarked'],drop_first=True)



df_train = pd.concat([df_train,sex,embark],axis=1)

df_test  = pd.concat([df_test ,sex,embark],axis=1)



df_train.drop(['Sex','Embarked'],axis=1,inplace=True)

df_test.drop(['Sex','Embarked'],axis=1,inplace=True)
imputer  = SimpleImputer()

scaler   = StandardScaler()

clf      = LogisticRegression()

pipe     = make_pipeline(imputer,scaler,clf)
features = df_train.drop('Survived',axis=1).columns



X,y   = df_train[features], df_train['Survived']

df_test.fillna(df_test.mean(),inplace=True)
pipe.fit(X,y)

y_pred = pd.DataFrame(pipe.predict(df_test))



y_pred['Survived'] = y_pred[0]

y_pred.drop(0,axis=1,inplace=True)

y_pred['PassengerId'] = df_test['PassengerId']



y_pred.to_csv('titanic_pred_logistic.csv',index=False)