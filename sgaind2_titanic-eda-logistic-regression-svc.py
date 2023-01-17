# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#Perform necessary imports: pandas, numpy, seaborn, sklearn, matplotlib.pyplot
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read the data to data variable
data = pd.read_csv('/kaggle/input/titanic/train.csv')
#first 5 rows in data
data.head()
#information of all features
data.info()
#dropping cabin column as it has large number of missing values
data = data.drop(['Cabin'], axis = 1)
#dropping name of the passenger, the passengerID and the ticket number will be unique for every single entry, so these features will not help us in predicting the correct output
data = data.drop(['Name','Ticket', 'PassengerId'], axis = 1)
#isolate output label and drop it from data
y = data[['Survived']]
data = data.drop(['Survived'], axis = 1)
#first 5 rows in data after cleaning
data.head()
#describe the numberical features in the data 
data.describe()
# create box plots of the 5 features seen above
data.Pclass.plot(kind='box')
plt.show()
data.Age.plot(kind='box')
plt.show()
data.SibSp.plot(kind='box')
plt.show()
data.Parch.plot(kind='box')
plt.show()
data.Fare.plot(kind='box')
plt.show()
# create violin plots of the 5 features seen above
sns.violinplot(y = 'Pclass', data=data )
plt.show()
sns.violinplot(y = 'Age', data=data )
plt.show()
sns.violinplot(y = 'SibSp', data=data )
plt.show()
sns.violinplot(y = 'Parch', data=data )
plt.show()
sns.violinplot(y = 'Fare', data=data )
plt.show()
#one hot encoding the Embarked Feature
OHE_Embarked = pd.get_dummies(data[['Embarked']])
pd.DataFrame([[OHE_Embarked.Embarked_C.sum(),OHE_Embarked.Embarked_Q.sum(),OHE_Embarked.Embarked_S.sum()]], columns = ['C','Q','S']).plot(kind='bar')
plt.show()
#one hot encoding the Sex feature
OHE_Sex = pd.get_dummies(data[['Sex']])
pd.DataFrame([[OHE_Sex.Sex_female.sum(),OHE_Sex.Sex_male.sum()]], columns = ['Female','Male']).plot(kind='bar')
plt.show()
#adding OHE columns to main data
data[['Embarked_C','Embarked_Q','Embarked_S']] = OHE_Embarked
data[['Sex_female','Sex_male']] = OHE_Sex
data = data.drop(['Embarked','Sex'], axis = 1)
data.head()
#split the data using an 80:20 random split 

train = data.sample(frac = 0.8, random_state = 20)
test = data.drop(train.index)
y_test=y.drop(train.index)
y_train = y.iloc[train.index]
print(train.shape)
print(test.shape)
print(y_train.shape)
print(y_test.shape)
#calculate mean of age column for training set
age_mean = train.Age.mean()
train.Age = train.Age.fillna(age_mean)
test.Age = test.Age.fillna(age_mean)
print(train.info())
print(test.info())
#instantiate a logistic regression model
logistic_regression_model = linear_model.LogisticRegression(max_iter  =10000, tol = 0.0001, random_state = 34)
#fit the model on the training set
logistic_regression_model.fit(train,y_train)
#predict using the test set and return the accuracy
logistic_regression_model.score(test,y_test)
#instantiate a support vector classifier model
svc = svm.SVC(kernel='linear')
#fit the model on the training set
svc.fit(train,y_train)
#predicted output for test data
prediction = svc.predict(test)
#calculated accuracy based on the original and predicted test value
acc = accuracy_score(y_test, prediction)
print ('\nAccuracy:', acc)
