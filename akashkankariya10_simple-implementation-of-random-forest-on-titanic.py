import numpy as np
import pandas as pd 
data=pd.read_csv('../input/train.csv')
data.head()
features=['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
data=data[features]
data.head()
data.info()
data.fillna(data.mean(),inplace=True)
data.head()
data['Cabin'].fillna(0,inplace=True)
y=data
j=0
for i in data['Cabin']:
    if(i!=0):
        data['Cabin'][j]=1
    j=j+1
data.head()
data.info()
data=pd.get_dummies(data)
data.head()
data.drop(['Sex_female','Cabin_1'],axis=1,inplace=True)
X=data.drop(['Survived'],axis=1)
y=data['Survived']
y.head()
X.head()
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.33, random_state=42)
train_X.head()
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(train_X,train_y)
pred=model.predict(test_X)
from sklearn import metrics
score=metrics.accuracy_score(test_y,pred)
score
