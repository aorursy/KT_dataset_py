# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Check the versions of libraries



# Python version

import sys

print('Python: {}'.format(sys.version))

# scipy

import scipy

print('scipy: {}'.format(scipy.__version__))

# numpy

import numpy

print('numpy: {}'.format(numpy.__version__))

# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))

# pandas

import pandas

print('pandas: {}'.format(pandas.__version__))

# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
# Load libraries

from pandas import read_csv

from pandas.plotting import scatter_matrix

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from scipy.stats import norm

import seaborn as sns

import matplotlib.pyplot as plt

import re as re
# Load dataset

url = "/kaggle/input/titanic/train.csv"

df = read_csv(url, header = 0)
#Data description

df.info()
df.head()
df.describe()
#Data visualization and transformation per variable

#PassengeriD

df['PassengerId'].head(10)
#Given no relevance of this variable we drop it from the df

df.drop(['PassengerId'], axis=1, inplace=True)
#Survived -> this is the variable to estimate

sns.barplot(x="Survived", data=df)
df.describe()['Survived']
#Evaluating how each of the other variables behaves versus _Survived_

#Pclass

df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot(x="Pclass", y="Survived", data=df)
#The higher the class, the most likely the passenger survived
#Sex

df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
sns.barplot(x="Sex", y="Survived", data=df)
df['Sex'] = df['Sex'] == 'male'
#SibSp

df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
sns.barplot(x="SibSp", y="Survived", data=df)
#Parch

df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
sns.barplot(x="Parch", y="Survived", data=df)
#FamilySize

#New variable counting family size using _Parch_ and _SibSp_

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df[["FamilySize", "Survived"]].groupby(['FamilySize'], as_index=False).mean()
sns.barplot(x="FamilySize", y="Survived", data=df)
#IsAlone

#New dummy variable taking 1 is passenger was on board alone

df['IsAlone'] = 0

df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
sns.barplot(x="IsAlone", y="Survived", data=df)
#Ticket

#Given no relevance of this variable we drop it from the df

df.drop(['Ticket'], axis=1, inplace=True)
#Embarked

#Fill missing values with mode

df['Embarked'] = df['Embarked'].fillna('S')

df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x="Embarked", y="Survived", data=df)
#Fare

sns.distplot(df['Fare'], fit = norm)
#Tranform _Fare_ to a normal distribution

df['Fare'] = np.log1p(df['Fare'])

sns.distplot(df['Fare'], fit = norm)
#And now transforming into categorical variable

df['FareGroup'] = pd.qcut(df['Fare'], 7, labels=['A', 'B', 'C', 'D', 'E', 'F', 'G'])

df[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean()
sns.barplot(x="FareGroup", y="Survived", data=df)
#Dropping orginal _Fare_ column

df.drop(['Fare'], axis=1, inplace=True)
#Cabin

#Transforming in binary variable, i.e. 1 if passenger was in a cabin

df['InCabin'] = ~df['Cabin'].isnull()
sns.barplot(x="InCabin", y="Survived", data=df)

plt.show()
#Dropping original _cabin_ column

df.drop(['Cabin'], axis=1, inplace=True)
#Age

#Given missing values, we group the variable into 8 categories

df["Age"] = df["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)
sns.barplot(x="AgeGroup", y="Survived", data=df)

plt.show()
#Dropping original _age_ column

df.drop(['Age'], axis=1, inplace=True)
#Name

#From this variable we extract the first part, whether the person as Mr, Miss, Mrs

df['Name'].head(10)
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



df['Title'] = df['Name'].apply(get_title)



pd.crosstab(df['Title'], df['Sex'])
#Categorizing into most common names

df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace('Ms', 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')



df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
sns.barplot(x="Title", y="Survived", data=df)

plt.show()
#Dropping original _Name_ column

df.drop(['Name'], axis=1, inplace=True)
#Correlation matrix

correlation_matrix = df.corr()

correlation_matrix



plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(correlation_matrix);
#Transforming categorical columns into binary variables

cols = ['Pclass', 'Embarked', 'FareGroup', 'AgeGroup', 'Title']

titanic_categorical = df[cols]

titanic_categorical = pd.concat([pd.get_dummies(titanic_categorical[col], prefix=col) for col in titanic_categorical], axis=1)

titanic_categorical.head()

df = pd.concat([df[df.columns[~df.columns.isin(cols)]], titanic_categorical], axis=1)

df.head()
#Training and Test

from sklearn.model_selection import train_test_split



X = df.drop('Survived', 1)

y = df.Survived



X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.30, random_state=42)
X.head()
# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn

results = []

names = []

for name, model in models:

	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

	results.append(cv_results)

	names.append(name)

	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#Conclusion

#Linear Discriminant Analysis is the best method for predicting the survivals in the titanic using this dataset