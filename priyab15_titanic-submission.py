import numpy as np 
import pandas as pd 
import seaborn as sns # Beautiful Graphs
sns.set_style('dark') # Set graph styles to 'dark'

import matplotlib.pyplot as plt # Normal ploating graphs
# show graphs in this notebook only 
%matplotlib inline

import plotly.express as px # For interactive plots

# ignore  the warning
import warnings  
warnings.filterwarnings('ignore') 


traindf = pd.read_csv('./../input/titanic/train.csv')
traindf.head()
testdf = pd.read_csv('./../input/titanic/test.csv')
testdf.head()
traindf.shape

testdf.shape
traindf.info()
testdf.info()
remove = ['Name','Ticket','Cabin']
traindf = traindf.drop(remove,axis = 1)

testdf = testdf.drop(remove,axis = 1)
traindf['Age'] = traindf['Age'].fillna(traindf['Age'].mean())
testdf['Age'] = testdf['Age'].fillna(testdf['Age'].mean())
traindf.info()
testdf.info()
traindf['Embarked'].value_counts()
traindf['Embarked'] = traindf['Embarked'].fillna('S')
testdf['Fare'] = testdf['Fare'].fillna(testdf['Fare'].mean())
traindf['Sex'].value_counts()
traindf['Sex'] = traindf['Sex'].replace({'male': 0, 'female': 1})
testdf['Sex'] = testdf['Sex'].replace({'male': 0, 'female': 1})
traindf['Embarked'].value_counts()
traindf['Embarked'] = traindf['Embarked'].replace ({ 'C' : 1, 'S' : 2, 'Q': 3})
testdf['Embarked'] = testdf['Embarked'].replace ({ 'C' : 1, 'S' : 2, 'Q': 3})
traindf.info()
testdf.info()
plt.figure(figsize=(10,15))
sns.heatmap(traindf.corr(),cmap = 'Blues', annot = True, linewidths = 1, fmt = '.1f')
fig = plt.gcf()
plt.show()
fig = px.bar(traindf.Survived.value_counts())
fig.show()
fig = px.bar(traindf.groupby(['Survived']).count())
fig.show()
fig = px.histogram(traindf,x = 'Survived', y = 'Pclass', color = 'Pclass')
fig.show()
fig = px.bar(traindf, x = 'Sex', y = 'Survived', color = 'Sex')
fig.show()
plt.figure(figsize=(10, 7))

sns.barplot(x = 'Parch', y= 'Survived', data= traindf)
plt.title("Parch and Survived Graph")

plt.show()
xtrain = traindf.drop(['PassengerId', 'Survived'], axis = 1)
ytrain = traindf['Survived']
xtest = testdf.drop(['PassengerId'], axis = 1)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
pred.shape
accu = model.score(xtrain,ytrain)
print( "Model Prediction Score:", (accu * 100).round(2))
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier()
model1.fit(xtrain,ytrain)
pred1 = model1.predict(xtest)
pred1.shape
accu1 = model1.score(xtrain,ytrain)
print( "Model Prediction Score:", (accu1 * 100).round(2))

dict = { 'PassengerId' : testdf['PassengerId'], 'Survived' : pred1}
new = pd.DataFrame(dict, )
new.shape
y = new.to_csv('./my_new_submission.csv', index=False)
print("successful")