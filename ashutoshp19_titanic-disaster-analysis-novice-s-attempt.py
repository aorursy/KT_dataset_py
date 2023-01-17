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
# Importing the essential libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Importing the Training Dataset



data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_train.head()
# Importing the Training Dataset



data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

data_test.head()
data_train.describe()
data_train.info()
data_test.info()
# Grouping data acc to Sex and displaying the mean of total people that survived.



data_train[['Sex','Survived']].groupby('Sex').mean()
# Grouping data acc to Passenger Class and displaying the mean of total people that survived.



data_train[['Pclass','Survived']].groupby('Pclass').mean()
# Grouping data acc to Parent/Child(ren) and displaying the mean of total people that survived.



data_train[['Parch','Survived']].groupby('Parch').mean()
# Grouping data acc to Siblings/Spouse and displaying the mean of total people that survived.



data_train[['SibSp','Survived']].groupby('SibSp').mean()
sns.barplot(x=data_train.Parch, y=data_train.Survived)
sns.barplot(x=data_train.Pclass, y=data_train.Survived)



# Seems like the Passenger Class is relevat to who survived by a small margin at least.
sns.barplot(x=data_train.Sex, y=data_train.Survived)



# This was obvious.
sns.barplot(x=data_train.Embarked, y=data_train.Survived)



# Not that big a deal, but still let's keep this.
plt.figure(figsize=(14,7))

sns.heatmap(data_train.corr(), annot = True, cmap = 'coolwarm')



# Not much to derive from this. No strong correlations seen.

# Lets do some feature Engineering to get better dataset before we fit the models.
print(data_train.isnull().sum())

print('='*50)

print(data_test.isnull().sum())
# Replace the NaN values in the Age column of training data with the median, as it makes sense to not use mean or other such values in case of Age.

data_train.Age.fillna(data_train.Age.median(), inplace = True)

# Replace the NaN values in the Age column of TEST data with the median of the Age colum on the TRAINING data.

data_test.Age.fillna(data_train.Age.median(), inplace = True)



# Replace the NaN values of the Fare column of the TEST data with the mean of the column values from the TRAINING data.

data_test.Fare.fillna(data_train.Fare.mean(), inplace = True)



# Drop the Embarked rows with NaN values as there are only two of them and it makes sense to do so as they are categorical.

data_train.dropna(subset = ['Embarked'], inplace = True)
# Drop the Cabin column entirely as it has a lot of missing values and it MAY not really be important if a person has a cabin or not.

data_train.drop(['Cabin'], axis = 1, inplace = True)

data_test.drop(['Cabin'], axis = 1, inplace = True)
data_train.head()
print(data_train.isnull().sum())

print('='*50)

print(data_test.isnull().sum())



# Better.
# Lets make a Size_of_family colum to get a combined result of the number of people a certain person was travelling with.

data_train['Size_of_family'] = data_train['SibSp'] + data_train['Parch'] + 1

data_train[['Size_of_family','Survived']].groupby('Size_of_family').mean()
# Same for the test dataset.

data_test['Size_of_family'] = data_test['SibSp'] + data_test['Parch'] + 1
# Splitting the passengers into age groups. 

# Age groups: '<=12': Child. '<=18': Teenager, '<=40': Adult, '<=60': 'Middle ages', '>60': 'Elderly'

data_train.loc[data_train['Age'] <= 12, 'Age'] = 0

data_train.loc[(data_train['Age'] > 12) & (data_train['Age'] <= 18), 'Age'] = 1

data_train.loc[(data_train['Age'] > 18) & (data_train['Age'] <= 40), 'Age'] = 2

data_train.loc[(data_train['Age'] > 40) & (data_train['Age'] <= 60), 'Age'] = 3

data_train.loc[data_train['Age'] > 60, 'Age'] = 4
data_train['Age'] = data_train['Age'].astype(int)
# Again for the test dataset.

data_test.loc[data_test['Age'] <= 12, 'Age'] = 0

data_test.loc[(data_test['Age'] > 12) & (data_test['Age'] <= 18), 'Age'] = 1

data_test.loc[(data_test['Age'] > 18) & (data_test['Age'] <= 40), 'Age'] = 2

data_test.loc[(data_test['Age'] > 40) & (data_test['Age'] <= 60), 'Age'] = 3

data_test.loc[data_test['Age'] > 60, 'Age'] = 4

data_test['Age'] = data_test['Age'].astype(int)
data_train.describe()
# Splitting Fare in groups.

data_train.loc[ data_train['Fare'] <= 10, 'Fare'] = 0

data_train.loc[(data_train['Fare'] > 10) & (data_train['Fare'] <= 20), 'Fare'] = 1

data_train.loc[(data_train['Fare'] > 20) & (data_train['Fare'] <= 40), 'Fare'] = 2

data_train.loc[(data_train['Fare'] > 40) & (data_train['Fare'] <= 70), 'Fare'] = 3

data_train.loc[(data_train['Fare'] > 70) & (data_train['Fare'] <= 100), 'Fare'] = 4

data_train.loc[(data_train['Fare'] > 100) & (data_train['Fare'] <= 200), 'Fare'] = 5

data_train.loc[(data_train['Fare'] > 200) & (data_train['Fare'] <= 350), 'Fare'] = 6

data_train.loc[ data_train['Fare'] > 350, 'Fare'] = 7

data_train['Fare'] = data_train['Fare'].astype(int)
data_test.loc[ data_test['Fare'] <= 10, 'Fare'] = 0

data_test.loc[(data_test['Fare'] > 10) & (data_test['Fare'] <= 20), 'Fare'] = 1

data_test.loc[(data_test['Fare'] > 20) & (data_test['Fare'] <= 40), 'Fare'] = 2

data_test.loc[(data_test['Fare'] > 40) & (data_test['Fare'] <= 70), 'Fare'] = 3

data_test.loc[(data_test['Fare'] > 70) & (data_test['Fare'] <= 100), 'Fare'] = 4

data_test.loc[(data_test['Fare'] > 100) & (data_test['Fare'] <= 200), 'Fare'] = 5

data_test.loc[(data_test['Fare'] > 200) & (data_test['Fare'] <= 350), 'Fare'] = 6

data_test.loc[ data_test['Fare'] > 350, 'Fare'] = 7

data_test['Fare'] = data_test['Fare'].astype(int)
# Converting the Embarked column values to numerical.

data_train['Embarked'] = data_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data_test['Embarked'] = data_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# Converting the Sex column values to numerical.



data_train['Sex'] = data_train['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

data_test['Sex'] = data_test['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
data_train
# Droppiing columns nt deemed important.



data_train.drop(['PassengerId', 'Name', 'Ticket' ], axis = 1, inplace = True)

data_test.drop(['Name', 'Ticket' ], axis = 1, inplace = True)
# Final Training Data

data_train
# Final Test Data



data_test
plt.figure(figsize=(14,7))

sns.heatmap(data_train.corr(), annot = True, cmap = 'coolwarm')
# Splitting the deendent variables from the rest of the data.



X_train = data_train.drop(['Survived'], axis = 1)

y_train = data_train['Survived']

X_test = data_test.drop(['PassengerId'],axis = 1)
# Fitting the model.



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
# Predicting the values.

predict1 = classifier.predict(X_test)



# Getting the accuracy score.

print("Logistic regression accuracy {:.2F}".format(classifier.score(X_train, y_train)*100))
# Fitting the model.



from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
# Predicting the values.

predict2 = classifier.predict(X_test)



# Getting the accuracy score.

print("KNN accuracy {:.2F}".format(classifier.score(X_train, y_train)*100))
# Fitting the model.



from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)
# Predicting the values.

predict3 = classifier.predict(X_test)



# Getting the accuracy score.

print("SVC accuracy {:.2F}".format(classifier.score(X_train, y_train)*100))
# Fitting the model.



from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
# Predicting the values.

predict4 = classifier.predict(X_test)



# Getting the accuracy score.

print("Kernel SVC accuracy {:.2F}".format(classifier.score(X_train, y_train)*100))
# Fitting the model.



from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
# Predicting the values.

predict5 = classifier.predict(X_test)



# Getting the accuracy score.

print("Gaussian Naive Bayes accuracy {:.2F}".format(classifier.score(X_train, y_train)*100))
# Fitting the model.



from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
# Predicting the values.

predict6 = classifier.predict(X_test)



# Getting the accuracy score.

print("Decision Tree accuracy {:.2F}".format(classifier.score(X_train, y_train)*100))
# Fitting the model.



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
# Predicting the values.

pedict7 = classifier.predict(X_test)



# Getting the accuracy score.

print("Random Forest accuracy {:.2F}".format(classifier.score(X_train, y_train)*100))
output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predict6})

output.to_csv('my_submission.csv', index=False)