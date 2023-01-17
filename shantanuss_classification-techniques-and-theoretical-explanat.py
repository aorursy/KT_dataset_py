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
import pandas as pd

 

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

train = pd.read_csv("../input/sex-input-toy-data/input.csv")

test = pd.read_csv("../input/test-class/test.csv")

train_df = pd.DataFrame(train)

# Input feature for training the classifier

X_train = train_df[['height','weight','foot_size']]

X_train
# output label class



y_train = train_df[['sex']]

y_train


# Input data is continous in nature, so we need to use gaussian NB classifier. 

# For Gaussian,  we need to compute the mean and variance of the input data. 



train.describe()
# model

clf = GaussianNB()



# cross validation of training dataset

kfold = KFold(n_splits=4, random_state=100)



# model fitting

clf.fit(X_train, y_train)



results = cross_val_score(clf, X_train, y_train, cv=kfold)

results.mean()
pred = clf.predict(test)

print (test," " , "belongs to class: ",pred)   # this gives the class : female as output. 
# function to calculate gaussian probability density function



from math import sqrt

from math import pi

from math import exp



total_entries =  y_train.count()



# define custom function to calculate gaussian probability for continuous input features. 

def calculate_gaussian_prob(x, mean , stdev):

    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))

    return (1 / (sqrt(2 * pi) * stdev)) * exponent



calculate_gaussian_prob(1.0, 1, 1)
# prob_male = float((y_train.sex.values == 'male').sum() / total_entries)

# prob_female = float((y_train.sex.values == 'female').sum() / total_entries)



# prob_male, prob_female
# test

# h_mean = X_train['height'].mean()

# w_mean = X_train['weight'].mean()

# f_mean = X_train['foot_size'].mean()

# h_std = X_train['height'].std()

# w_std = X_train['weight'].std()

# f_std = X_train['foot_size'].std()
# posterior_male = prob_male*(calculate_gaussian_prob()
iris_data = pd.read_csv("../input/iris-flower-dataset/Iris_flower_dataset.csv")

iris_data.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline
# scatter plot
# shape

print(iris_data.shape)
iris_data.isnull().sum()
print(iris_data.info())
iris_data['Species'].unique()
iris_data["Species"].value_counts()
iris_data.describe()
iris_data.columns
cols = iris_data.columns

features = cols[0:4]

labels = cols[4]

print(features)

print(labels)
# Seperating labels and features from the main dataset.



X = iris_data.iloc[:,:-1].values

y = iris_data.iloc[:,-1 ].values

X
# using train_test_split to seprate training and validation data



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 10)

X_train, y_train
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# Model

clf = KNeighborsClassifier(n_neighbors=10)



# cross validation of training dataset

kfold = KFold(n_splits=4, random_state=100)



# model fitting

clf.fit(X_train, y_train)



results = cross_val_score(clf, X_train, y_train, cv=kfold)

results.mean()


# prediction

y_pred = clf.predict(X_valid)

y_pred
# Summary of the predictions made by the classifier

print(classification_report(y_valid, y_pred))

print(confusion_matrix(y_valid, y_pred))
# Accuracy score



print('accuracy is',accuracy_score(y_pred,y_valid))
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



# model

clf_g = GaussianNB()



# cross validation of training dataset

kfold = KFold(n_splits=4, random_state=100)



# model fitting

clf_g.fit(X_train, y_train)



results = cross_val_score(clf_g, X_train, y_train, cv=kfold)

results.mean()

# prediction

y_pred = clf_g.predict(X_valid)



# Summary of the predictions made by the classifier

print(classification_report(y_valid, y_pred))

print(confusion_matrix(y_valid, y_pred))

# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_valid))

# Decision Tree

from sklearn.tree import DecisionTreeClassifier



clf_d = DecisionTreeClassifier()



# cross validation of training dataset

kfold = KFold(n_splits=4, random_state=100)



# model fitting

clf_d.fit(X_train, y_train)



results = cross_val_score(clf_g, X_train, y_train, cv=kfold)

results.mean()

clf_d.fit(X_train, y_train)



y_pred = clf_d.predict(X_valid)



# Summary of the predictions made by the classifier

print(classification_report(y_valid, y_pred))

print(confusion_matrix(y_valid, y_pred))

# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_valid))