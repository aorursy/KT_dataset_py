import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
dataset=pd.read_csv('../input/titanic/train.csv')
dataset.describe()
dataset.isnull().any()
dataset=dataset.fillna(method='ffill')
test=pd.read_csv('../input/titanic/test.csv')
za=test['PassengerId']
test.describe()
dataset.Sex=dataset.Sex.map({'male':0,'female':1} )
dataset.Embarked=dataset.Embarked.map({'S':1,'C':2,'Q':3})
plt.subplot2grid((3,4),(0,0))
dataset.Survived[(dataset.Sex==0)&(dataset.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Rich Men Survived")

plt.subplot2grid((3,4),(0,1))
dataset.Survived[(dataset.Sex==0)&(dataset.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("poor Men Survived")

plt.subplot2grid((3,4),(0,2))
dataset.Survived[(dataset.Sex==1)&(dataset.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Rich Women Survived")

plt.subplot2grid((3,4),(0,3))
dataset.Survived[(dataset.Sex==1)&(dataset.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("poor Women Survived")

x=dataset.iloc[:,[2,4,11]].values
y=dataset.iloc[:,1].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
print (x[0:10, :])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x,y)

test.isnull().any()
test=test.fillna(method='ffill')
test.Sex=test.Sex.map({'male':0,'female':1} )
test.Embarked=test.Embarked.map({'S':1,'C':2,'Q':3})
test.drop('Name',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)
test.drop('Fare',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
test.drop('PassengerId',axis=1,inplace=True)
test.drop('Age',axis=1,inplace=True)
test.drop('SibSp',axis=1,inplace=True)
test.drop('Parch',axis=1,inplace=True)
test.info()
test.describe()

y_pred = classifier.predict(test)
print(y_pred)

subm=pd.DataFrame({"PassengerId":za,"survived":y_pred})
subm.to_csv("titanic.csv",index=False)    


