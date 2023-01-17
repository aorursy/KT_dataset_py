# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)



#%reset -f

dataset = pd.read_csv('../input/train.csv')

# Display propertice

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

dataset.describe()



#NAN check

print(dataset.isnull().sum())



#Duplicate check

print(sum(dataset.duplicated(dataset.columns)))



#Pandas profile

#import pandas_profiling

#profile = pandas_profiling.ProfileReport(dataset)

#print(profile)

abs(dataset.skew())

abs(dataset.kurt())



# Variance check - if variance less than threshold remove them

#from sklearn.feature_selection import VarianceThreshold

#constant_filter = VarianceThreshold(threshold=0.5)

#constant_filter.fit(dataset)

#print(dataset.columns[constant_filter.get_support()])

#constant_columns = [column for column in dataset.columns if column not in dataset.columns[constant_filter.get_support()]]

#dataset.drop(labels=constant_columns, axis=1, inplace=True)



X= dataset.drop('price_range', axis=1)

y= dataset['price_range']



X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=7)



from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier



# The "accuracy" scoring is proportional to the number of correct classifications

clf_rf_4 = RandomForestClassifier() 

rfecv = RFECV(estimator=clf_rf_4, step=1, cv=10, scoring='accuracy')   #5-fold cross-validation

rfecv = rfecv.fit(X_train, y_train)



print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X_train.columns[rfecv.support_])





cl = [col for col in X_train.columns if col not in X_train.columns[rfecv.support_]]



dataset.drop(cl, axis=1, inplace=True)



X= dataset.drop(['price_range'], axis=1)

y= dataset['price_range']



#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X = sc.fit_transform(X)

#Splitting the Dataset into Training set and Test Set

X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=7)

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier



# Classification

models =[]

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))







for name,model in models:

    cv_res = model_selection.cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy')

    cv_predict = model_selection.cross_val_predict(model,X_train,y_train,cv=10)

    print(f"{name}: {confusion_matrix(y_train, cv_predict)}")

    print(f"{accuracy_score(y_train,cv_predict)}")



for name,model in models:

    model.fit(X_train,y_train)

    pres= model.predict(X_test)

    print(f"{name}: {confusion_matrix(pres, y_test)}")

    print(f"{accuracy_score(pres,y_test)}")

    print(classification_report(y_test,pres))

## Training the RandomForestClassifier model

model = RandomForestClassifier()

model.fit(X_train, y_train)
##################Test Data#################3



dataset = pd.read_csv('../input/test.csv')



#NAN check

print(dataset.isnull().sum())

#Duplicate check

print(sum(dataset.duplicated(dataset.columns)))
dataset.drop(cl, axis=1, inplace=True)



X= dataset.drop(['id'], axis=1)



#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X = sc.fit_transform(X)

test_pred = model.predict(X)
test_pred