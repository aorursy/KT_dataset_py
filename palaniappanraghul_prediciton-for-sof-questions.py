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
#loading the dataset

data = pd.read_csv("/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv")

print(data.head())
#Checking for null values

print(data.isnull().sum())
#input features

x = data[["Title","Body","Tags","CreationDate"]]

print(x.head())
#target feature

y = data["Y"]

print(y.head())
#label encoding for converting labels into numeric form

from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()  

x= x.apply(label_encoder.fit_transform)

print(x.head())
y= label_encoder.fit_transform(y)

print(y)
#target feature visualization

import matplotlib.pyplot as plt

plt.hist(y)

plt.title("Question Classification Rating") 
#train-test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#Training ML model and finding its accuracy

from sklearn.ensemble import ExtraTreesClassifier

nb = ExtraTreesClassifier(random_state=0, n_estimators=100,  criterion='gini')

nb.fit(x_train, y_train)

y_nb = nb.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_nb))
#Training ML model and finding its accuracy

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier( criterion='gini', splitter='best')

dtc.fit(x_train, y_train)

y_dtc = dtc.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_dtc))
#Training ML model and finding its accuracy

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0, n_estimators=100,  criterion='gini')

clf.fit(x_train, y_train)

y_clf = clf.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_clf))
#Decoding the label encoded values from random forest classifier prediction

z=label_encoder.inverse_transform(y_clf)

print(z)