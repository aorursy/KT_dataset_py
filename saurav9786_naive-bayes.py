# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading the dataset

pima_indians_df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
pima_indians_df.describe().T
pima_indians_df['Outcome'].value_counts()
pima_indians_df['SkinThickness'].value_counts(0)
pima_indians_df['Insulin'].value_counts()
# Data Cleaning

import random



pima_indians_df['SkinThickness']=pima_indians_df['SkinThickness'].replace(0,random.randrange(30, 40))



pima_indians_df['Insulin']=pima_indians_df['Insulin'].replace(0,random.randrange(30, 140))



pima_indians_df.head(10)







array = pima_indians_df.values

X = array[:,0:8] # select all rows and first 7 columns which are the attributes

Y = array[:,8]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties

test_size = 0.15 # taking 70:30 training and test set

seed = 7  # Random numbmer seeding for reapeatability of the code

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


model = GaussianNB()

model.fit(X_train, Y_train)

print(model)

# make predictions

expected = Y_test

predicted = model.predict(X_test)

# summarize the fit of the model

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))