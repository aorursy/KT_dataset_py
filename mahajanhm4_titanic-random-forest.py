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
import numpy as np #For math

import pandas as pd #For handling data

import matplotlib.pyplot as plt #For visualizing and graphing data

import seaborn as sns # for plotting and visualizing data

import warnings

warnings.filterwarnings('ignore') #for ignoring any version specific warnings

from sklearn.preprocessing import LabelEncoder #Encodes categorical variables into numbers

from sklearn.preprocessing import OneHotEncoder #Encodes the result of the labelencoder 

from sklearn.preprocessing import StandardScaler #Scales the data with appropriate range
#reading new titanic dataset

titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv') #pandas reads the csv file and makes a variable

print("Train dimensions = {}".format(titanic_train.shape)) #the .shape attribute returns a tuple denoting the size of the dataset

titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv') #pandas reads the csv file and makes a variable

print("Test dimensions = {}".format(titanic_test.shape)) #the .shape attribute returns a tuple denoting the size of the dataset
#lets have a look at our data

titanic_train.head(5)
#finding null value status of the data

print("***Train dataset*** ")

print(titanic_train.isna().sum()) #the method .isna() returns a boolean array .sum() over it returns total number of trues in the array

print("***Test dataset*** ")

print(titanic_test.isna().sum()) #the method .isna() returns a boolean array .sum() over it returns total number of trues in the array
# Finding and replacing the null values with average in dataset for column age

titanic_train['Age'].fillna(titanic_train['Age'].mean(), inplace = True) #replaces blank fields with mean age inplace

print("Null Values in Age for training data: {}".format(titanic_train['Age'].isna().sum()))

titanic_test['Age'].fillna(titanic_test['Age'].mean(), inplace = True) #replaces blank fields with mean age inplace

print("Null Values in Age for test data: {}".format(titanic_test['Age'].isna().sum()))
#cabin is irrelevant data, it adds to noise and most cannot be recovered hence we have decided to remove it

titanic_train = titanic_train.drop(columns='Cabin')

titanic_test = titanic_test.drop(columns='Cabin')
# Re-evaluating null value situation

print("Train dataset: ")

print(titanic_train.isna().sum())

print("Test dataset: ")

print(titanic_test.isna().sum())
# Finding and replacing the null values with average in dataset for column age

titanic_train['Fare'].fillna(titanic_train['Fare'].mean(), inplace = True) #replaces blank fields with mean age inplace

print("Null Values in Fare for training data: {}".format(titanic_train['Fare'].isna().sum()))

titanic_test['Fare'].fillna(titanic_test['Fare'].mean(), inplace = True) #replaces blank fields with mean age inplace

print("Null Values in Fare for test data: {}".format(titanic_test['Fare'].isna().sum()))
# Now our data is clean

print("***Train dataset***")

print(titanic_train.isna().sum())

print("***Test dataset***")

print(titanic_test.isna().sum())
#passengerID ,name, fare and embarked are not helpful hence we remove them 

cols_removed = ['PassengerId','Name', 'Ticket']

titanic_train = titanic_train.drop(columns=cols_removed)

titanic_test = titanic_test.drop(columns= cols_removed)
titanic_train = titanic_train.dropna()
#before plotting our heatmap we need to encode our categorical variables

titanic_train['Sex'] = titanic_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

titanic_test['Sex'] = titanic_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
titanic_train['Embarked'] = titanic_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

titanic_test['Embarked'] = titanic_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#Correlation analysis

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(titanic_train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
titanic_train_X = titanic_train.iloc[:,1:]

titanic_train_y = titanic_train.Survived
# before we scale we need to onehotencode the data for categorical variables 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [6])

titanic_train_X = onehotencoder.fit_transform(titanic_train_X).toarray()
#the process of smoothening is to scale all the values in the column  from one specific range 

# this helps the ml algorithms to not get influenced by sizes of the parameters

from sklearn.preprocessing import MinMaxScaler

Xscaler = MinMaxScaler()

titanic_train_X = Xscaler.fit_transform(titanic_train_X)

#now our data has been completely transformed into a usable form
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(titanic_train_X, titanic_train_y, test_size = 0.2, random_state = 0)
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 50, min_samples_split=16, min_samples_leaf=

                                    1, max_features='auto', oob_score=True,max_depth=10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

actual = y_test

predicted =  classifier.predict(X_test)

results = confusion_matrix(actual, predicted)

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(actual, predicted)) 

print('Report : ')

print(classification_report(actual, predicted))
titanic_test.head()
# before we scale we need to onehotencode the data for categorical variables 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder_test = OneHotEncoder(categorical_features = [6])

titanic_test_new = onehotencoder.fit_transform(titanic_test).toarray()
#the process of smoothening is to scale all the values in the column  from one specific range 

# this helps the ml algorithms to not get influenced by sizes of the parameters

newscaler = MinMaxScaler()

titanic_test_new = newscaler.fit_transform(titanic_test_new)

#now our data has been completely transformed into a usable form
test_y_pred = classifier.predict(titanic_test_new)
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv') #pandas reads the csv file and makes a variable

titanic_test.columns
PID = list(titanic_test.PassengerId)
submit_dict = {'PassengerId': PID, 'Survived': list(test_y_pred)}

submit = pd.DataFrame.from_dict(submit_dict)
submit
submit.to_csv('my_submission.csv', index=False)