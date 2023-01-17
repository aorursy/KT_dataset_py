#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected= True)
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
#load data
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
submit=pd.read_csv("../input/gender_submission.csv")
import warnings
warnings.filterwarnings('ignore')
train.info()
train.head()
#Bar chart of no. of people survived
train.Survived.value_counts().sort_index().plot.bar()
#Bar chart of Age
train.Age.value_counts().head(25).plot.bar()
#Bar chart of Pclass
train.Pclass.value_counts().plot.bar()
#Bar chart of Sibsp
train.SibSp.value_counts().plot.bar()
#Bar chart of Cabin
train.Cabin.value_counts().head(15).plot.bar()
#line plot of Age
train.Age.value_counts().sort_index().plot.line()
#Histogram of Fare
train[train['Fare']<100]['Fare'].plot.hist()
#Scatter plot of Fare with Pclass
train[train['Fare']<100].sample(100).plot.scatter(x='Fare',y='Pclass')
#Scatter plot of Age vs Survived
train[train['Age']<30].sample(100).plot.scatter(x='Age',y='Survived')
#Hexbin of Age vs Survived
train[train['Age']<30].plot.hexbin(x='Age',y='Survived',gridsize=10)
#Stacked bar plot
train_survived=train.groupby(['Survived','Embarked']).mean()[['Pclass','SibSp']]
train_survived.plot.bar(stacked=True)
#Stacked bar plots of Parch and SibSp grouped by Survived, Embarked, Pclass
train_survived=train.groupby(['Survived','Embarked','Pclass']).mean()[['Parch','SibSp']]
train_survived.plot.bar(stacked=True)
#Stacked bar plot
train_survived=train.groupby(['Survived','Sex']).mean()[['Parch','SibSp']]
train_survived.plot.bar(stacked=True)
#count plot of Embarked
sns.countplot(train['Embarked'])
#count plot of Pclass
sns.countplot(train['Pclass'])
#KDE plot of Fare
sns.kdeplot(train.query('Fare<200').Fare)
#KDE plot of Age
sns.kdeplot(train.query('Age<30').Age)
#Dist plot of Fare
sns.distplot(train[train['Fare']<100]['Fare'],bins=10,kde=False)
#Dist plot of Age
sns.distplot(train[train['Age']<30]['Age'],bins=50,kde=False)
#facetgrid
df=train
g=sns.FacetGrid(df,col="Embarked",col_wrap=3)
g.map(sns.kdeplot,"Age")
df=train
g=sns.FacetGrid(df,col="SibSp",row="Pclass")
g.map(sns.kdeplot,"Age")
train["family_size"] = 0
test["family_size"] = 0
    
train["family_size"] = train["SibSp"] + train["Parch"]
test["family_size"] = test["SibSp"] + test["Parch"]

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2 
train["Sex"][train["Sex"] == "female"] = 0
train["Sex"][train["Sex"] == "male"] = 1 
    
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2 
test["Sex"][test["Sex"] == "female"] = 0
test["Sex"][test["Sex"] == "male"] = 1 
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
dependent_variables = ['Pclass', 'Sex', 'Age', 'SibSp','Embarked','Parch','Fare', 'family_size']


train_x = train[dependent_variables].values
test_x = test[dependent_variables].values

train_y = train["Survived"].values


random_forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

model = random_forest.fit(train_x, train_y)
prediction = model.predict(test_x)

