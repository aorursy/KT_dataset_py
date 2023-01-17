import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sb

sb.set_style('darkgrid')
train = pd.read_csv('../input/titanic/train.csv')
train.head()
train['Title'] = [i.split('.')[0].split(',')[1] for i in train['Name']]
sb.countplot(x='Survived',hue='Sex',data = train )

plt.title('Survival Vs Sex')
sb.countplot(x='Survived',hue='Embarked',data=train)

plt.title('Survival of Pasengers according to Embarked')
sb.countplot(x='Survived',hue='Pclass',data=train)

plt.title('Survival vs Pclass')
sb.barplot(x='Sex',y='Fare',hue='Embarked',data=train)

plt.title('Fare Vs Sex according to Embarked')
sb.barplot(y='Age',x='Sex',hue='Pclass',data=train)

plt.title('Survival of Gender as per Pclass')
plt.figure(figsize=(10,8))

sb.lineplot(x='Age',y='Fare',hue='Embarked',data=train)

plt.title('Fare of the People Embarked according to their Age')
plt.figure(figsize=(12,8))

sb.lineplot(x='Age',y='Fare',hue='Sex',data=train)

plt.title('Fare of the people according to Sex')
plt.figure(figsize=(10,8))

sb.lineplot(x='Age',y='Fare',hue='Pclass',data=train)

plt.title('Fare of the People Embarked according to their Age')
sb.FacetGrid(train,col='Survived',row='Title').map(plt.scatter,'Age','Fare',edgecolor="w")
sb.FacetGrid(train,row='Sex',col='Survived').map(plt.scatter,'Age','Fare',edgecolor="w",color='g')
sb.lmplot(x='Age',y='Fare',col='Survived',hue='Sex',data=train)

plt.title('Survial Vs Age Vs Fare')
sb.lmplot(x="Age", y="Fare", col="Sex", hue="Pclass", data=train)

plt.title('Fare Vs Age according to Pclass')
sb.FacetGrid(train, row  ='Sex',col='Survived',margin_titles= True).map(plt.hist ,

'Age', color='steelblue')  
plt.figure(figsize=(8,8))

sb.countplot(x='Title',hue='Sex',data = train)

plt.xticks(rotation = 45)