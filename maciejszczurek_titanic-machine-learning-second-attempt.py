# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# pandas
import pandas as pd
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
test_dataset = pd.read_csv('/kaggle/input/titanic/test.csv')
combined = [train_dataset, test_dataset]
# Getting the data

train_dataset.head()
train_dataset.info()
print('_'*40)
test_dataset.info()
# displaying information -- we need to correct null values
train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# predicting that class mattered in terms of survival
train_dataset[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# predicting that sex mattered in terms of survival
g = sns.FacetGrid(train_dataset, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# predicting that age mattered in therms of survival -- especially very young children
train_dataset = train_dataset.drop(['PassengerId', 'Name', 'SibSp','Parch','Ticket', 'Cabin', 'Fare', 'Embarked'], axis=1)
test_dataset = test_dataset.drop(['Name', 'SibSp','Parch','Ticket', 'Cabin', 'Fare', 'Embarked'], axis=1)
combined = [train_dataset, test_dataset]
# removing data that wont be analysed

train_dataset.head()
for dataset in combined:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

guess_ages = np.zeros((2,3))
for dataset in combined:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
    
train_dataset['AgeBand'] = pd.cut(train_dataset['Age'], 5)
train_dataset[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combined:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    
train_dataset = train_dataset.drop(['AgeBand'], axis=1)
combined = [train_dataset, test_dataset]

train_dataset.head()
# converting the data into one that's easier to analyse
X_train = train_dataset.drop("Survived", axis=1)
Y_train = train_dataset["Survived"]
X_test  = test_dataset.drop("PassengerId", axis=1).copy()

# preparing datasets as ready for learning
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# logarithmic regression to check for correlations between the data and viewing the accuracy 
coeff_df = pd.DataFrame(train_dataset.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# checking which datapoints correlate the most
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

# using the random forest model to generate our predictions and the accuracy of it
submission = pd.DataFrame({
        "PassengerId": test_dataset["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)
#saving our data redy for submission

# Whole work was heavily based on this tutorial https://www.kaggle.com/startupsci/titanic-data-science-solutions