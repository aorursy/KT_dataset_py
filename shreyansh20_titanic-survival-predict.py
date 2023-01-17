import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy

train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
train_data.isnull().sum()
median=train_data['Age'].median()
median
train_data
train_data['Embarked']=train_data['Embarked'].fillna('S')
train_data['Age']=train_data['Age'].fillna(median)

passengers=train_data['PassengerId']

data=train_data.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)

data.isnull().sum()
test_data.isnull().sum()
passengers1=test_data['PassengerId']
df2=pd.DataFrame({'PassengerId':passengers1})
df2
test_data['Age']=test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].median())

data1=test_data.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)

Y=data['Survived']
data=data.drop(['Survived'],axis=1)
data.head(10)
features=['Sex','Embarked']
%matplotlib inline
ax = data['Embarked'].value_counts().plot(kind='bar',figsize=(7,4),color='r')
plt.show()
%matplotlib inline
ax = data['Sex'].value_counts().plot(kind='bar',figsize=(7,4),color='y')
plt.show()
plt.plot(data['Fare'],data['Age'],'bo')
plt.show()
%matplotlib inline
ax = data['Parch'].value_counts().plot(kind='bar',figsize=(7,4),color='g')
plt.show()
for i in features:
    le=preprocessing.LabelEncoder()
    le.fit(data[i])
    le.transform(data[i])
    data[i]=le.fit_transform(data[i])

rf=RandomForestClassifier()
rf.fit(data,Y)
for j in features:
    LE=preprocessing.LabelEncoder()
    LE.fit(data1[j])
    LE.transform(data1[j])
    data1[j]=LE.fit_transform(data1[j])
pred=rf.predict(data1)
df = pd.DataFrame({'Survived': pred})
df
result = pd.concat([df2, df], axis=1, join='inner')
 
result.to_csv('gender__submission.csv')
ld=LinearDiscriminantAnalysis()
ld.fit(data,Y)
pred1=ld.predict(data1)
df3 = pd.DataFrame({'Survived': pred1})
result1 = pd.concat([df2, df3], axis=1, join='inner')

result.to_csv('gendersubmission.csv')
