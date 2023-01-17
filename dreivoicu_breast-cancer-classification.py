# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data Viz

import matplotlib.pyplot as plt

import seaborn as sns # statistical data vizualization



# Sklearn

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import classification_report, confusion_matrix



# Config

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')



# shape of the data

print(data.shape)
# printing the first rows of the data

data.head()
# describing the data

data.describe()
# getting the info of the data

data.info()
# checking if the dataset contains any NULL Values

null_counts = data.isna().sum()

null_counts = null_counts[null_counts > 0]



print(null_counts)
# checking for unique values of features

data.nunique()
# removing the id column as it is unique

data = data.drop('id', axis = 1)



print(data.shape)
# removing the last column as it is empty

data = data.drop('Unnamed: 32', axis = 1)



print(data.shape)
# interesting features to plot

target_vars = [

    'radius_mean',

    'texture_mean',

    'area_mean',

    'perimeter_mean',

    'smoothness_mean'

]

sns.pairplot(

    data, 

    hue = 'diagnosis', 

    vars = target_vars

)
sns.countplot(data.diagnosis)
sns.lmplot(

    'area_mean',

    'smoothness_mean',

    hue ='diagnosis',

    data = data,

    fit_reg=False

)
# plots the given feature with respect to the target feature

def plot_feature(df, feature, target_feature):

    fig = sns.FacetGrid(df, hue=target_feature, aspect=4)

    fig.map(sns.kdeplot, feature, shade= True)

    fig.add_legend()
plot_feature(data, 'radius_mean', 'diagnosis')
plot_feature(data, 'texture_mean', 'diagnosis')
plot_feature(data, 'area_mean', 'diagnosis')
plot_feature(data, 'perimeter_mean', 'diagnosis')
plot_feature(data, 'smoothness_mean', 'diagnosis')
fig = plt.figure(figsize = (20, 10))



corr = data.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(

    corr,

    mask = mask,

    cmap = 'RdYlGn', 

    annot = True, 

    fmt=".1f")
# label encoding of the dependent variable

le = LabelEncoder()

data.diagnosis = le.fit_transform(data.diagnosis)



data.diagnosis.value_counts()
# splitting the dependent and independent variables from the dataset

X = data.iloc[:,1:]

y = data.iloc[:,0]



print(X.shape)

print(y.shape)
# splitting the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



print(X_train.shape)

print(y_train.shape) 



print(X_test.shape)

print(y_test.shape)
# data normalization

sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
# Support Vector Machine

from sklearn.svm import SVC



# HYPER PARAMETER TUNING FOR SVM

# using grid search to find the best parameters for svm



param = {

    'C': [0.1, 1, 10, 100],

    'kernel':['linear', 'rbf'],

    'gamma' :[0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]

}

grid_svc = GridSearchCV(

    SVC(),

    param_grid = param,

    scoring = 'accuracy',

    cv = 10

)
# Feeding the training data to the grid model

# and also finding the best parameters



grid_svc.fit(X_train, y_train)

print("Best Parameters: ", grid_svc.best_params_)

print("Best Accuarcy: ", grid_svc.best_score_)
# creating a new SVC model with these best parameters

model = grid_svc.best_estimator_

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(classification_report(y_test, y_pred))

print("Testing accuarcy :", model.score(X_test, y_test))



cm = confusion_matrix(y_test, y_pred)

sns.heatmap(

    cm,

    annot = True,

    cmap = 'copper',

    fmt='d'

)