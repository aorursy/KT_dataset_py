import pandas as pd
import numpy as np
import sklearn
titanic_data = pd.read_csv('../input/titanic/train.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

titanic_data['Sex'].replace(['female','male'],[0,1],inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=[('Embarked')])
titanic_data.drop(columns=['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
titanic_data['Age'].fillna(np.mean(titanic_data['Age']),inplace=True)

predictors = []
for col in titanic_data.columns:
    predictors.append(col)
target = predictors.pop(0)

train_data, test_data, train_sln, test_sln = train_test_split(titanic_data[predictors], titanic_data[target], test_size=0.2, random_state=2)
%%time

forest_vals = []
for i in range(1,1500,150):
    forest = RandomForestClassifier(random_state = 8,n_estimators=i)
    forest.fit(train_data,train_sln)
    dt_prediction = forest.predict(test_data)
    forest_vals.append(metrics.accuracy_score(test_sln,dt_prediction))

import matplotlib.pyplot as plt
%matplotlib inline
plt.suptitle('Accuracy as n_estimators changes',fontsize=18)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.plot(range(1,1500,150),forest_vals,'r-',label='accuracy using n_estimators')
plt.legend(loc='best', shadow=True)
plt.show()
%%time

forest_vals = []
for i in range(1,30):
    forest = RandomForestClassifier(random_state = 8,n_estimators=700,max_depth=i)
    forest.fit(train_data,train_sln)
    dt_prediction = forest.predict(test_data)
    forest_vals.append(metrics.accuracy_score(test_sln,dt_prediction))

import matplotlib.pyplot as plt
%matplotlib inline
plt.suptitle('Accuracy as max_depth changes',fontsize=18)
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.plot(range(1,30),forest_vals,'r-',label='accuracy using max_depth')
plt.legend(loc='best', shadow=True)
plt.show()
# Make predictions for test data
def convert(d):
    d['Sex'].replace(['female','male'],[0,1],inplace=True)
    d = pd.get_dummies(d, columns=[('Embarked')])
    d.drop(columns=['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
    d['Age'].fillna(np.mean(d['Age']),inplace=True)
    d['Fare'].fillna(np.mean(d['Fare']),inplace=True)
    return d

test = convert(pd.read_csv('../input/titanic/test.csv'))
test.isna().any()
forest = RandomForestClassifier(random_state = 8,n_estimators=700,max_depth=14)
forest.fit(titanic_data[predictors],titanic_data['Survived'])
dt_prediction = forest.predict(test)
predicted = pd.DataFrame(data={'PassengerId':range(892,1310),'Survived':dt_prediction}).to_csv('results.csv', index = False)
