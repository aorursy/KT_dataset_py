# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection
df=pd.read_csv("/kaggle/input/titanic/train.csv")

df_test=pd.read_csv("/kaggle/input/titanic/test.csv")

combined_dataset = [df , df_test]
df_test.isnull().sum()
df.isnull().sum()
df.drop(['PassengerId'  ,'Ticket' , 'Embarked' ,'Cabin'] ,axis =1, inplace=True)

df_test.drop(['Ticket' , 'Embarked' ,'Cabin'] ,axis =1, inplace=True)

df.dropna(inplace=True)

df_test['Age'].fillna(df_test['Age'].mean() , inplace =True)

df_test['Fare'].fillna(method = 'ffill',inplace=True)
sns.countplot(x=df.Survived , hue = df.Pclass)
sns.countplot(x=df.Survived , hue = df.Sex)
for dataset in combined_dataset:

    dataset.Sex = pd.get_dummies(dataset.Sex , drop_first=True)

    dataset.loc[:,'Title'] = dataset.loc[:,'Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset.loc[:,'Title'] = dataset.loc[:,'Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset.loc[:,'Title'] = dataset.loc[:,'Title'].replace('Mlle', 'Miss')

    dataset.loc[:,'Title'] = dataset.loc[:,'Title'].replace('Ms', 'Miss')

    dataset.loc[:,'Title'] = dataset.loc[:,'Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combined_dataset:

    dataset.loc[:,'Title'] = dataset.loc[:,'Title'].map(title_mapping)

    dataset.loc[:,'Title'] = dataset.loc[:,'Title'].fillna(0)
df
for dataset in combined_dataset:

    dataset['Age_band'] = pd.cut(df.loc[:,'Age'],5)

for dataset in combined_dataset:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']
df
for dataset in combined_dataset:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combined_dataset:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
df
for dataset in combined_dataset:

    dataset.drop(['Name' ,'SibSp' , 'Parch' , 'Age_band'] ,axis =1 , inplace=True)

df
df_test
X_train = df.drop('Survived' , axis =1)

y_train = df.Survived
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))

models.append(('DescisionTree', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('LSVC' , LinearSVC(dual=False)))

models.append(('PCT' , Perceptron()))

models.append(('SGD' , SGDClassifier()))

models.append(('RNDFOREST' , RandomForestClassifier()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10 ) #random_state=seed

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring )

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# # boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names ,rotation =40)

plt.show()
df_testing = df_test.drop('PassengerId' ,axis =1)

random_forest = DecisionTreeClassifier()

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(df_testing)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)
submission