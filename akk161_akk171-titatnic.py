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
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
gender_submissoin=pd.read_csv("../input/gender_submission.csv")
print(train.shape, test.shape)
# Checking the class balance 
print("Event_Rate",round(train.Survived.sum()/train.Survived.count(),2))
# Lets build some relevent feature and do some EDA
# 1. Checking various column type and Missing values etc 
for c in train.columns:
    print (c, "----", train[c].dtype)
# Lets check columns one by one 
print(train.groupby(by='Pclass')['Survived'].sum())
print(train.groupby(by='Pclass')['Survived'].count())
#class_wise_Survival_rate
Fisrt = round(train.groupby(by='Pclass')['Survived'].sum()[1]/train.groupby(by='Pclass')['Survived'].count()[1],2)
Second = round(train.groupby(by='Pclass')['Survived'].sum()[2]/train.groupby(by='Pclass')['Survived'].count()[2],2)
Third = round(train.groupby(by='Pclass')['Survived'].sum()[3]/train.groupby(by='Pclass')['Survived'].count()[3],2)
print("Ist :",Fisrt, "2nd  :", Second, "3rd :", Third)
# Does anyone variable have mising values 
round(train.isna().sum()/train.shape[0],5)
# As cabin has very high missing rate and hence removing cabin for now 
# We would have used the MODE value in case of categorigal variable # df = df.fillna(df['Label'].value_counts().index[0])
# Because Age have very low missing rate lets us impute missing value using any of the regression algorithms or for simplicity
# lets keep average or median as imputation value 
# import matplotlib
# import missingno as msno
# %matplotlib inline
# missingdata_df = train.columns[train.isnull().any()].tolist()
# msno.matrix(train[missingdata_df],figsize=(6,4))
# msno.bar(train[missingdata_df], color="blue", log=True, figsize=(6,4))
# fill missing values with mean column values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='median')
train['Age'] = imputer.fit_transform(train[['Age']])
train.groupby(by='Embarked')['Survived'].sum()
train.groupby(by='Embarked')['Fare'].mean()
train[train['Embarked'].isna()]
train['Embarked'].fillna('C',inplace=True)
# x=pd.DataFrame(train.groupby(by='Cabin')['Ticket'].count())
# CabinMem = x.reset_index()
# CabinMem.columns=['Cabin','Cabin_Member']
# train=train.merge(CabinMem, on='Cabin',how='left')
#Adding family size as a variable 
train['familyZize']= train['SibSp']+train['Parch']
# Apply all feature engineering to test Data 
#Adding family size as a variable 
test['familyZize']= test['SibSp']+test['Parch']
train.head()
# train['Cabin_Member'].fillna('0', inplace= True)
# len(train['Cabin'].unique())
dummies = pd.get_dummies(train[['Sex','Embarked']])
train = pd.concat([dummies, train], axis=1)
X_train= train[['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Pclass','Age', 'SibSp','Parch','Fare','familyZize']]
Y_train= train[['Survived']]
print(X_train.shape,Y_train.shape)
from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier()
rf=RandomForestClassifier(n_estimators=500, n_jobs=-1, verbose=True, oob_score=True)
rf.fit(X_train,Y_train)
rf.score(X_train,Y_train)
# Checking missing value in Test data 
round(test.isna().sum()/test.shape[0],5)
print(test.shape)
# test[test['Cabin'].isna()].shape
# test['Cabin_Member'].fillna('0', inplace= True)
# test.head()
dummies = pd.get_dummies(test[['Sex','Embarked']])
test = pd.concat([dummies, test], axis=1)
# fill missing values with mean column values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='median')
test['Age'] = imputer.fit_transform(test[['Age']])
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='median')
test['Fare'] = imputer.fit_transform(test[['Fare']])
test.head()
X_test= test[['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Pclass','Age', 'SibSp','Parch','Fare','familyZize']]
# X_test.apply(lambda x: (x==np.inf).sum())
# X_test.apply(lambda x: (x.isna()).sum())
for c in X_test.columns:
    print(c, "----", X_test[c].dtype, "----", max(X_test[c]), min(X_test[c]))
X_test=X_test[['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'familyZize']]
for c in X_test.columns:
    print(c, "----", X_train[c].dtype, "----", max(X_train[c]), min(X_train[c]))
rf.predict_proba(X_train)
X_test.describe()
X_train.describe()
xtest=X_test.fillna(X_test.mean())
Predict_Test = pd.DataFrame(rf.predict(xtest), columns={'Survived'})
test[['PassengerId']].head()
Predict_Test = pd.merge(test[['PassengerId']],Predict_Test,how='left',left_index=True, right_index=True)
Predict_Test.head()
Predict_Test.to_csv('akk171_Submission.csv')
