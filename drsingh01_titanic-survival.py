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
data_test =  pd.read_csv ('../input/titanic/test.csv')

data_test.head()
data_train =  pd.read_csv ('../input/titanic/train.csv')

data_train.head()
import matplotlib.pyplot as plt

plt.matshow(data_train.corr())

plt.show()
data_train.describe()
# Get column data types

data_train.dtypes
# count the number of NaN values in each column

data_train.isnull().sum(axis = 0)
# Subfunctions for data cleaning

#fill missing values with mean of column values



from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):



    def __init__(self):

        self.fill = 0

        return

    def fit(self, X, y=None):

        

        self.fill = pd.Series([X[c].value_counts().index[0]

        # if not X and not series, error

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)                              

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)



dataframeClean = DataFrameImputer().fit_transform(data_train)
data_train1 = dataframeClean.copy()
data_train1.isnull().sum(axis = 0)
data_train1.head()
data_train1.iloc[:,4] 
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data_train1.iloc[:, 4] = labelencoder.fit_transform(data_train1.iloc[:, 4])

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data_train1.iloc[:, 3] = labelencoder.fit_transform(data_train1.iloc[:, 3])

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data_train1.iloc[:, 8] = labelencoder.fit_transform(data_train1.iloc[:, 8])
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data_train1.iloc[:, 10] = labelencoder.fit_transform(data_train1.iloc[:, 10])
data_train12 = data_train1.copy()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data_train1.iloc[:, 11] = labelencoder.fit_transform(data_train1.iloc[:, 11])
data_train1.head(10)
test_1.head()
test12 = test_1.iloc[:, :].values
print(test12)
X = data_train1.iloc[:, data_train.columns != 'Survived'].values

y = data_train1.iloc[:,data_train.columns == 'Survived'].values

print(X)
print(y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

len(X_train)
len(y_train)
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(y_pred)
print(y_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 
print(accuracy_score(y_test, y_pred))
# count the number of NaN values in each column

data_test.isnull().sum(axis = 0)
dataframeCleanTest = DataFrameImputer().fit_transform(data_test)
test_1 = dataframeCleanTest.copy()
test_1.head()
test_1.iloc[:, 3] = labelencoder.fit_transform(test_1.iloc[:, 3])

test_1.iloc[:, 2] = labelencoder.fit_transform(test_1.iloc[:, 2])

test_1.iloc[:, 9] = labelencoder.fit_transform(test_1.iloc[:, 9])

test_1.iloc[:, 10] = labelencoder.fit_transform(test_1.iloc[:, 10])

test_1.iloc[:, 7] = labelencoder.fit_transform(test_1.iloc[:, 7])
test_1.head(12)
# Get column data types

test_1.dtypes
print(test12)
# Predicting the Test set results

y_pred12 = classifier.predict(test_1)



print(y_pred12)


len(test_1)
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test_1['PassengerId'],'Survived':y_pred12})



#Visualize the first 5 rows

submission.head(20)
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)