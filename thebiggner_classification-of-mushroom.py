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
# Importing libraries

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
# Importing machine learning algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score
# Importing dataset

mushroom = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
# Getting a look at the data.

mushroom.head()
mushroom.info()
mushroom.shape
for col in mushroom.columns:

    print('{}:{}'.format(col, mushroom[col].unique()))
mushroom.groupby(by='class').agg({'class':'count'}).plot(kind='bar')
# Separating the target from the data

y = mushroom['class']

X = mushroom.iloc[:, 1:]



# Splitting in training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=132)
# Imputing missing value

imputer = SimpleImputer(missing_values='?', strategy='most_frequent')

X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_test)
# Transforming X_train and X_test in dataframes

X_train = pd.DataFrame(X_train, columns=X.columns)

X_test = pd.DataFrame(X_test, columns=X.columns)



# Adding column in train and test sets

X_train['label'] = 1

X_test['label'] = 0



# Concatnating both datasets

data = pd.concat([X_train, X_test], axis=0)

 

# Applying LabelEncoder on the dataset

encoder = LabelEncoder()

encoded_data = data.iloc[:,:22].apply(encoder.fit_transform)

encoded_data = pd.DataFrame(encoded_data, columns=X.columns)

encoded_data['label'] = data['label']

encoder1 = LabelEncoder()

y_train = encoder1.fit_transform(y_train)

y_test = encoder1.transform(y_test)
# Separating train and test set

X_train = encoded_data[encoded_data['label']==1]

X_test = encoded_data[encoded_data['label']==0]

X_train.drop('label', axis=1, inplace=True)

X_test.drop('label', axis=1, inplace=True)
tree = DecisionTreeClassifier(random_state=1)

tree.fit(X_train, y_train)

prediction = tree.predict(X_test)

print('Accuracy of the model is : {}'.format(accuracy_score(prediction, y_test)))
forest = RandomForestClassifier(random_state=1)

forest.fit(X_train, y_train)

prediction = forest.predict(X_test)

print("Accuracy of the model is : {}".format(accuracy_score(prediction, y_test)))
print('Precision of model : {}'.format(precision_score(prediction, y_test)))