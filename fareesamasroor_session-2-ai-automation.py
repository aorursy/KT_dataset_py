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
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import pickle

from sklearn.model_selection import train_test_split

loan = pd.read_csv("../input/loan-data-set/loan_data_set.csv")

Loan_ = loan.drop(columns=['Loan_ID'])



one_hot_encoded_training = pd.get_dummies(Loan_)

data= one_hot_encoded_training



data[data==np.inf]=np.nan

data.fillna(data.mean(), inplace=True)



X = data.iloc[:, :21]

y = data.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train, y_train)
result = model.score(X_test, y_test)

print("Accuracy: %.3f%%" % (result*100.0))
Loan_issued = model.predict(X.head())

Loan_issued