# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
test=pd.read_csv('../input/titanic/test.csv')
train=pd.read_csv('../input/titanic/train.csv')
train.head(5)
train.info()
test.info()
test.shape, train.shape
train.describe()
train.isna().sum()
print(test.isnull().sum().sort_values(ascending=False))
train.groupby('Sex')[['Survived']].mean()
train['Sex'].value_counts().plot.bar()
plt.title('Sex')
plt.xticks(rotation=360)
plt.ylabel('Counts');
train['Pclass'].value_counts().plot.bar()
plt.title('Pclass')
plt.xticks(rotation=360)
plt.ylabel('Counts');
train['Embarked'].value_counts().plot.bar()
plt.title('Embarked')
plt.xticks(rotation=360)
plt.ylabel('Counts');
#Plot the survival rate of each class.
sns.barplot(x='Pclass', y='Survived', data=train)
males = train[train['Sex'] == 'male']
females = train[train['Sex'] == 'female']

survived_males = males['Survived'].value_counts()
survived_females = females['Survived'].value_counts().sort_values(ascending=True)

n_groups = 2
index = np.arange(n_groups)

width = 0.3

plt.bar(np.arange(len(survived_males)), survived_males, width=width, label='Male')
plt.bar(np.arange(len(survived_females)) + 0.3, survived_females, width=width, label='Female', color='orange')
plt.xticks(index + 0.15, ('0', '1'), rotation=360)
plt.title('Survived by Sex')
plt.xlabel('Survived')
plt.ylabel('Counts')
plt.legend()
plt.figtext(0.90, 0.01, '0 = No, 1 = Yes', horizontalalignment='right');
# Missing value treatment
data = [train, test]

for dataset in data:
    mean = train["Age"].mean()
    std = test["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train["Age"].astype(int)
train["Age"].isnull().sum()
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train,test]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train['Embarked'].describe()
train.info()
#Print the unique values in the columns
print(train['Sex'].unique())
print(train['Embarked'].unique())
train.isna().sum()
train.head(5)
#Encoding categorical data values (Transforming object data types to integers)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#Encode sex column
train.iloc[:,2]= labelencoder.fit_transform(train.iloc[:,2].values)
#print(labelencoder.fit_transform(titanic.iloc[:,2].values))

#Encode embarked
train.iloc[:,7]= labelencoder.fit_transform(train.iloc[:,7].values)
#print(labelencoder.fit_transform(titanic.iloc[:,7].values))

#Print the NEW unique values in the columns
print(train['Sex'].unique())
print(train['Embarked'].unique())
train = train.dropna(subset =['Embarked'])
train.isna().sum()
train.info()
data = [train, test]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train, test]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)
genders = {"male": 0, "female": 1}
data = [train, test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
train['Ticket'].describe()
# Ticket attribute has 680 unique tickets,So we will drop it from the dataset.
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

train.head(5)
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
#Split the data into independent 'X' and dependent 'Y' variables
X = train.iloc[:, 1:8].values 
Y = train.iloc[:, 0].values
# Split the dataset into 80% Training set and 20% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#Create a function within many Machine Learning Models
def models(X_train,Y_train):
  

  #Using SVC method of svm class to use Support Vector Machine Algorithm
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear', random_state = 0)
  svc_lin.fit(X_train, Y_train)


  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
  print('[0]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  print('[1]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
  return svc_lin, forest
#Get and train all of the models
model = models(X_train,Y_train)