import pandas as pd

import requests

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
data=train.loc[:,['Pclass','Sex','Age','SibSp','Parch']]

target=train.loc[:,'Survived']
data.head()
data['Sex'].unique()
data.head()
sexd=data.loc[:,'Sex']

sexd=pd.get_dummies(sexd)
sexd.head()
data=data.join(sexd)
data.drop(columns='Sex',axis=1,inplace=True)
data.head()
testdata=test.loc[:,['Pclass','Sex','Age','SibSp','Parch']]

testdata.head()
sexdt=testdata.loc[:,'Sex']

sexdt=pd.get_dummies(sexdt)
sexdt.head()
testdata.drop(columns='Sex',inplace=True)

testdata=testdata.join(sexdt)

testdata.head()
from sklearn.preprocessing import Imputer



trainData = Imputer().fit_transform(data)

testData=Imputer().fit_transform(testdata)
from sklearn.svm import SVC

titanic=SVC()
titanic
titanic.fit(trainData,target)
pred=titanic.predict(trainData)
from sklearn.metrics import classification_report as clf

print(clf(target,pred))
predict=titanic.predict(testData)
print(predict)
id=test.loc[:,'PassengerId']
id.shape
result=pd.DataFrame(id)

result.shape
predict.shape
result['Survived']=predict
result.head()