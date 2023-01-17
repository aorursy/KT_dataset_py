# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import cufflinks as cf #make interactive plots with cufflinks

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#read the csv in a dataframe cnamed "pulsar.csv"

pulsar = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")



#Take a look at the head of the this datafram to see the columns

pulsar.head()
pulsar.columns
#Setup the X and the y (predict label) from the dataset

X=pulsar.drop('target_class', axis=1)

y=pulsar['target_class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)



from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)