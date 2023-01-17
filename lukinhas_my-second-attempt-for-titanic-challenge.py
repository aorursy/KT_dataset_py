# ANY CONSTRUCTIVE FEEDBACK IS WELCOME, I'M TRYING TO START A CAREER IN ANALYTICS AND DATA SCIENCE,
# SO ANY ADVICE LIKE WHAT TO STUDY, WHAT I CAN IMPROVE IN THIS KERNEL TO MAKE IT PROFESSIONAL AND
# OTHERS STUFFS IS VERY WELCOME GUYS.
# IMPORTING SOME LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# IMPORTING TRAIN AND TEST DATA

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# LETS TAKE A LOOK IN OUR DATA AND MAKE SOME ANALYSES

train.head()
train.info()
train.describe()
plt.figure(figsize = (12,6))
sns.heatmap( data = train.isnull(), cbar = False, cmap = 'inferno', yticklabels = False)
sns.set_style('whitegrid')

# WE NOTICE HERE SOME DATA MISSING. THE CABIN COLUMN HAS A LOT OF DATA MISSING, PROBABLY I'LL DISCARD THIS 
# COLUMN. FOR THE AGE COLUMN, I'LL REPLACE THE MISSING DATA FOR A MEAN VALUE.
#LETS MAKE SOME GRAPHICS AND UNDESTAND OUR DATA
plt.figure(figsize = (12,6))
sns.countplot(x = 'Survived', hue = 'Sex', data = train, palette = 'Pastel2')

# WE CAN SEE THAT THE NUMBER OF NON SURVIVED IS HIGHER AND MOST PART IS FROM MALE SEX.
plt.figure(figsize = (12,6))
sns.countplot(x = 'Survived', hue = 'Pclass', data = train, palette = 'inferno')

# THE BIGGEST PART OF NON SURVIVED IS COMPOSED BY THE THIRD CLASS AND THE HIGHEST NUMBER OF SURVIVED IS FROM 
# FIRST CLASS.
plt.figure(figsize = (12,6))
train['Age'].hist(bins = 50, color = 'red', alpha = 0.5)

# CHECKING THE DISTRIBUTION OF AGE.
plt.figure(figsize = (12,6))
sns.countplot(data = train, x = 'SibSp')

# THE BIGGEST PART OF THE PASSENGERS WENT TO TRAVEL ALONE. MAYBE BECAUSE THE MAJORITY WAS YOUNG, LETS CHECK.
plt.figure(figsize = (12,6))
train[train['SibSp']== 0]['Age'].hist(bins = 70, ALPHA = 0.6)

# WE CONFIRM THE HIPOTHESIS.
# LETS CHECK THE FARE COLUMN.
plt.figure(figsize = (12,6))
train['Fare'].hist(bins = 50, color = 'green', alpha = 0.5)
# LETS GIVE A ZOOM IN THE REGION OF MAJOR CONCENTRATION.
plt.figure(figsize = (12,6))
train[train['Fare']<70]['Fare'].hist(bins = 50, color = 'black', alpha = 0.5)
# THE VALUE OF THE TICKET CONCENTRATES BELOW OF 10 DOLLARS, PROBABLY IT IS THE COST FOR THIRD CLASS. LETS SEE
# THE PREDOMINANT CLASS
plt.figure(figsize = (12,6))
sns.countplot(train['Pclass'], palette = 'PuBuGn')
# HERE IS SOME INTERISTING GRAPHICS THAT I SAW IN KAGGLE BY MANAV SEHGAL.
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# NOW LETS WORK IN OUR DATA.
# THIS IS MY SECOND KERNEL FOR THIS CHALLENGE, NOW I'LL TRY TO USE THE NAME COLUMN AND THE PARCH COLUMN IN MY
# MODEL.
train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]
for dt in combine:
    dt['Title'] = dt.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])
for dt in combine:
    dt['Title'] = dt['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Differentiated')
    dt['Title'] = dt['Title'].replace('Mlle', 'Miss')
    dt['Title'] = dt['Title'].replace('Ms', 'Miss')
    dt['Title'] = dt['Title'].replace('Mme', 'Mrs')
dTitle = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Differentiated": 5}
for dt in combine:
    dt['Title'] = dt['Title'].map(dTitle)
train.head()
train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]
train.head()
dSex = {"male": 0, "female": 1}
for dt in combine:
    dt['Sex'] = dt['Sex'].map(dSex)
train.head()
train_Class1 = train[train.Pclass < 2]
train_Class1['Age'].mean()
train_Class2 = train[train.Pclass == 2]
train_Class2['Age'].mean()
train_Class3 = train[train.Pclass > 2]
train_Class3['Age'].mean()
def imput_age(col):
    Age = col[0]
    Class = col[1]
    
    if pd.isnull(Age):
        if Class == 1:
            return 38
        elif Class == 2:
            return 30
        else:
            return 25
    else:
        return Age
for dt in combine:
    dt['Age'] = dt[['Age', 'Pclass']].apply(imput_age, axis = 1)
plt.figure(figsize = (12,6))
sns.heatmap( data = train.isnull(), cbar = False, cmap = 'inferno')
sns.set_style('whitegrid')
train.dropna(inplace = True)
test.dropna(inplace  = True)
for dt in combine:
    dt['Embarked'] = dt['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()
test.head(10)
# Model, predict and solve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( train.drop('Survived', axis = 1), train['Survived'], test_size = 0.30)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print (classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
print (confusion_matrix (y_test, predictions))