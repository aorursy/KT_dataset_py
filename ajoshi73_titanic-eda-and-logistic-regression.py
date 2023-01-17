# Important Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt
# Reading training data
tdata = pd.read_csv('/kaggle/input/titanic/train.csv')

# First 5 rows of tdata 
tdata.head()
# Getting info of features
tdata.info()
# Seperating output label from the tdata data
y = tdata[['Survived']]
tdata = tdata.drop(['Survived'], axis = 1)
# Dropping Cabin feature since it has very less number of rows in comparison to other features
tdata = tdata.drop(['Cabin'], axis = 1)

# Removing unique features, these will not help in classification
tdata = tdata.drop(['Name','Ticket', 'PassengerId'], axis = 1)

# tdata after changes
tdata.head()
# Describing numerical features
tdata.describe()
# Creating histograms of the above 5 features

tdata.Pclass.plot(kind='hist')
plt.xlabel('Pclass')
plt.title('Distribution of Pclass')
plt.show()
tdata.Age.plot(kind='hist')
plt.xlabel('Age')
plt.title('Distribution of Age')
plt.show()
tdata.SibSp.plot(kind='hist')
plt.xlabel('SibSp')
plt.title('Distribution of SibSp')
plt.show()
tdata.Parch.plot(kind='hist')
plt.xlabel('Parch')
plt.title('Distribution of Parch')
plt.show()
tdata.Fare.plot(kind='hist')
plt.xlabel('Fare')
plt.title('Distribution of Fare')
plt.show()
# Applying One Hot Encoding to Sex Feature, providing numbered labels to male and female
New_S = pd.get_dummies(tdata[['Sex']])
New_S.head()
# Applying One Hot Encoding to Embarked Feature, providing numbered labels to character values
New_E = pd.get_dummies(tdata[['Embarked']])
New_E.head()
# Adding One hot encoded features to tdata nad droping prevoius features
tdata[['Sex_female','Sex_male']] = New_S
tdata[['Embarked_C','Embarked_Q','Embarked_S']] = New_E
tdata = tdata.drop(['Sex','Embarked'], axis = 1)

# updated tdata
tdata.head()
# Getting info of tdata
tdata.info()
# Filling remaining rows in Age feature with mean age value
mean_a = tdata.Age.mean()
tdata.Age = tdata.Age.fillna(mean_a)

# checking info
tdata.info()
#spliting the tdata by random split of 70:30

train = tdata.sample(frac = 0.7, random_state = 42)
test = tdata.drop(train.index)
y_test=y.drop(train.index)
y_train = y.iloc[train.index]
print(train.shape)
print(test.shape)
print(y_train.shape)
print(y_test.shape)
# Setting logistic regression model
logistic_regression_model = linear_model.SGDClassifier(loss = 'log', max_iter  =10000, tol = 0.0001, random_state = 30)
#model fit on the training set
logistic_regression_model.fit(train,y_train)
#prediction using the test set and returning the accuracy
acc = logistic_regression_model.score(test,y_test)
print('accuracy = ',acc*100,'%')
