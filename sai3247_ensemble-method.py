import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.dtypes
test.dtypes
train.isnull().sum().sort_values(ascending=False)
train.Cabin.fillna(train.Cabin.value_counts().idxmax(),inplace=True)
train.Age.fillna(train.Age.value_counts().idxmax(),inplace=True)
train.Embarked.fillna(train.Embarked.value_counts().idxmax(),inplace=True)
train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
test.Cabin.fillna(test.Cabin.value_counts().idxmax(),inplace=True)
test.Age.fillna(test.Age.value_counts().idxmax(),inplace=True)
test.Fare.fillna(test.Fare.value_counts().idxmax(),inplace=True)
test.isnull().sum().sort_values(ascending=False)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
intcols=train.select_dtypes(include=['int64'])
intcols1=intcols.apply(le.fit_transform)
floatcols=train.select_dtypes(include=['float64'])
objectcols=train.select_dtypes(include=['object'])
objectcols1=objectcols.apply(le.fit_transform)
train1=pd.concat([objectcols1,intcols1,floatcols],axis=1)
train1.dtypes
intcols=test.select_dtypes(include=['int64'])
objectcols=test.select_dtypes(include=['object'])
floatcols=test.select_dtypes(include=['float64'])
intcols1=intcols.apply(le.fit_transform)
objectcols1=objectcols.apply(le.fit_transform)
test1=pd.concat([intcols1,floatcols,objectcols1],axis=1)
test1.dtypes
y=train1.Survived
x=train1.drop(['Survived','PassengerId'],axis=1)
xtest=test1.drop('PassengerId',axis=1)
x.shape
xtest.shape
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(n_estimators=2000)
gbcmodel=gbc.fit(x,y)
gbcmodel.score(x,y)
predict=gbcmodel.predict(xtest)
predict
submission = pd.DataFrame(data={'PassengerId': (np.arange(len(predict)) + 1), 'Survived': predict})
submission.to_csv('gender_submission.csv', index=False)
submission.tail()  
