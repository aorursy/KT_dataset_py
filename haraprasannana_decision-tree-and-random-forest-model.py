#Importing necessary libraries

import pandas as pd

from sklearn import tree

from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import RandomizedSearchCV

from sklearn.base import TransformerMixin

% matplotlib inline
#Reading the files

dataset = pd.read_csv("../input/mushrooms.csv", na_values= '?', keep_default_na = False)
dataset.head()
# Checking the data types

dataset.info()
#Shape of dataset

dataset.shape
#checking null values

dataset.isnull().sum()
class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with mean of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)
#X = pd.DataFrame(dataset['stalk-root'])

dataset['stalk-root'] = DataFrameImputer().fit_transform(pd.DataFrame(dataset['stalk-root']))
#checking null values

dataset.isnull().sum()
labelencoder = LabelEncoder()

for col in dataset.columns:

    dataset[col] = labelencoder.fit_transform(dataset[col])

dataset.head()
# Defining x and y variables

x = dataset.iloc[:, 1:23].values

y = dataset.iloc[:, :1].values
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=123)
#Fitting classifier to training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier = classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
#Confusion Matrix

cm = metrics.confusion_matrix(y_test, y_pred)

cm
#Fitting classifier to training set

from sklearn.ensemble import RandomForestClassifier



classifier1 = RandomForestClassifier()

classifier1.fit(x_train, y_train.ravel())
y_pred = classifier1.predict(x_test)
#Confusion Matrix

cm = metrics.confusion_matrix(y_test, y_pred)

cm