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
# Importing necessary liraries:



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Importing data:



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Viewing the training data:



train.head(3)
# Viewing the test data: 



test.head(3)     #Only difference in terms of column with `train data` is absence of 'Survived' column.
# Info() method will help us mainly to identify the amount null values.



train.info()     #We have null values for Age, Cabin & Embarked.
# Test data:



test.info()       #Null values for Age, Fare & Cabin Column
# Getting information about the rows and column of train and test data set:



train.shape, test.shape
# For getting an overall statistics of the train data:



train.describe()
# 'O' is for getting a statistics of "Object Data Type":



train.describe(include=['O'])  
# Using seaborn's heatmap to explore the missing values:



sns.heatmap(train.isnull(), yticklabels = False, cbar = False)
train.isnull().sum()       #More easy way to count the number null values. 
# Visualizing Number of passengers Survived or Not:



sns.set(style = 'whitegrid')

sns.countplot(x = 'Survived', data = train)
# Visualizing Number of passengers Survived or Not based on Sex:



sns.countplot(x = 'Survived', hue = 'Sex', data = train)
# Visualizing Number of passengers Survived or Not based on Passenger Class:



sns.countplot(x = 'Survived', hue = 'Pclass', data = train, palette = 'rainbow')
# Getting an idea about the age of passengers:



sns.distplot(train['Age'].dropna(), kde = False, color = 'darkgreen', bins = 30)   #setting kde false to view histogram.
# Visualizing various price range of ticket: 



train['Fare'].hist(bins = 30, figsize = (8, 4))
# Finding the mean age of each passenger class:



plt.figure(figsize = (12, 8))

sns.boxplot(x = 'Pclass', y = 'Age', data = train)
# Setting the Missing Age values with Mean Age on the basis of Passenger Class:



def mean_age_finder(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
# Updating the Age column with apply() method:



train['Age'] = train[['Age', 'Pclass']].apply(mean_age_finder, axis = 1)
# Verifying if there is any null value in Age column:



train.isnull().sum()
# Now we will replace the two null values of 'Embarked' column.

# We will replace NaN values with most frequent value of this column: (Port from where the particular passenger was embarked/boarded): 



most_frequent_embarked = train['Embarked'].value_counts().index[0]      #Output: 'S'

train['Embarked'].fillna(most_frequent_embarked, inplace = True)
# Verifying if there is null values in Embarked column. Also we can see that Cabin column has a lot of null values. So we will drop it.



train.isnull().sum()
# Dropping `Cabin` column from train data:



train.drop('Cabin', axis = 1, inplace = True)
# Verifying if there is any null values remains in our train data:



# sns.heatmap(train.isnull(), yticklabels = False, cbar = False)     #Way 1

train.isnull().sum()                                                 #Way 2 (I prefer this method)
# Finding the mean age of each passenger class:



plt.figure(figsize = (12, 8))

sns.boxplot(x = 'Pclass', y = 'Age', data = test)
# Setting the Missing Age values with Mean Age on the basis of Passenger Class:



def mean_age_finder_test(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 42

        elif Pclass == 2:

            return 27

        else:

            return 24

    else:

        return Age
# Updating the Age column using apply() method:



test['Age'] = test[['Age', 'Pclass']].apply(mean_age_finder_test, axis = 1)
# Verifying if there is any null value in Age column:



test.isnull().sum()
#Replacing the missing value of 'Fare' column with mean value:



test['Fare'].fillna(test['Fare'].mean(), inplace = True)
# Dropping `Cabin` column from test data:



test.drop('Cabin', axis = 1, inplace = True)
# Verifying if there is any null value in test data:



test.isnull().sum()
# Verifying is there is any null values remains in our test data (Another Way):



sns.heatmap(test.isnull(), yticklabels = False, cbar = False)
# Setting drop_first = True is for preventing multicollinearity, as one column is opposite of another. 



sex = pd.get_dummies(train['Sex'], drop_first = True)

embarked = pd.get_dummies(train['Embarked'], drop_first = True)
# Removing Categorical Columns and added newly converted dummy columns:



train.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)

train = pd.concat([train, sex, embarked], axis = 1)
# Viewing train data after dropping and adding columns:



train.head()
# Creating dummy variables:



sex = pd.get_dummies(test['Sex'], drop_first = True)

embarked = pd.get_dummies(test['Embarked'], drop_first = True)
# Removing Categorical Columns and added newly converted dummy columns (test data):



test.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)

test = pd.concat([test, sex, embarked], axis = 1)
# Viewing test data after dropping and adding columns:



test.head()
X_train = train.drop(['PassengerId', 'Survived'], axis = 1)

y_train = train['Survived']

X_test = test.drop("PassengerId", axis=1).copy()
X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression(solver = 'liblinear')

logmodel.fit(X_train, y_train)



log_predictions = logmodel.predict(X_test)
accuracy = round(logmodel.score(X_train, y_train) * 100, 2)

print('Logistic Regression Accuracy: ', accuracy)
Submission = pd.DataFrame({

    "PassengerId": test['PassengerId'],

    "Survived": log_predictions

})



Submission.to_csv('Submission.csv', index = False)
# Viewing the submission file: 



Submission.head()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)      # Model's Accuracy depends on number of neighbors.

knn.fit(X_train, y_train)

knn_predctions = knn.predict(X_test)



knn_accuracy = round(knn.score(X_train, y_train) * 100, 2)

print('KNN Model Accuracy: ', knn_accuracy)
knnOut = pd.DataFrame({

    "PassengerId": test['PassengerId'],

    "Survived": knn_predctions

})



knnOut.to_csv('KNN.csv', index = False)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

dtree_predictions = dtree.predict(X_test)



dtree_accuracy = round(dtree.score(X_train, y_train) * 100, 2)

print('Decision Tree Model Accuracy: ', dtree_accuracy)
dtreeOut = pd.DataFrame({

    "PassengerId": test['PassengerId'],

    "Survived": dtree_predictions

})



dtreeOut.to_csv('dtree.csv', index = False)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100)

rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)



rfc_accuracy = round(rfc.score(X_train, y_train) * 100, 2)

print('Random Forest Model Accuracy: ', rfc_accuracy)
rfcOut = pd.DataFrame({

    "PassengerId": test['PassengerId'],

    "Survived": rfc_predictions

})



rfcOut.to_csv('rfc.csv', index = False)
from sklearn.svm import SVC

svc = SVC(gamma = 'auto')

svc.fit(X_train, y_train)

svc_predictions = svc.predict(X_test)



svc_accuracy = round(svc.score(X_train, y_train) * 100, 2)

print('SVC Accuracy: ', svc_accuracy)
svcOut = pd.DataFrame({

    "PassengerId": test['PassengerId'],

    "Survived": svc_predictions

})



svcOut.to_csv('rfc.csv', index = False)
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'K-Nearest Neighbors','Decision Tree', 'Random Forest', 'Support Vector Machines'],

    

    'Score': [accuracy, knn_accuracy, dtree_accuracy, 

              rfc_accuracy,  svc_accuracy]

    })



models.sort_values(by='Score', ascending=False)