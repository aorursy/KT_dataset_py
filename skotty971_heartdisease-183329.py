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
heart=pd.read_csv("../input/heart.csv")

heart.head()
heart.describe()
import matplotlib.pyplot as plt 

heart.hist(bins=50,figsize=(20,15))

plt.show()
#Columns Names

heart.columns
#Data Exploration

import seaborn as sns

heart.target.value_counts()

sns.countplot(x="target", data=heart, palette="bwr")

plt.show()
df=heart

countNoDisease = len(df[df.target == 0])

countHaveDisease = len(df[df.target == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
df.groupby('target').mean()
#Create Dummy Variable

#Logistic Regression

y = df.target.values

x_data = df.drop(['target'], axis = 1)





#Normalise Data



# Normalize So, the entire range of values of X from min to max are mapped to the range 0 to 1.Min-max normalisation is often known as feature scaling 

# where the values of a numeric range of a feature of data, i.e. a property, are reduced to a scale between 0 and 1. 

# Yi = [Xi - min(X)]/[max(X) - min(X)]

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values



#Visualisation of Normalized Data

import matplotlib.pyplot as plt 

x.hist(bins=50,figsize=(20,15))

plt.show()
#4 different automatic feature selection techniques:





# Feature Extraction with RFE

from pandas import read_csv

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

# load data



array = heart.values

X = array[:,0:13]

Y = array[:,13]

# feature extraction

model = LogisticRegression()

rfe = RFE(model, 3)

fit = rfe.fit(X, Y)

print(fit.n_features_)

print(fit.support_)

print(fit.ranking_)





#PCA

# Feature Extraction with PCA

import numpy

from pandas import read_csv

from sklearn.decomposition import PCA

array = heart.values

X = array[:,0:13]

Y = array[:,13]

# feature extraction

pca = PCA(n_components=3)

fit = pca.fit(X)

# summarize components

print (fit.explained_variance_ratio_)

print(fit.components_)



#FeatureImportance

from pandas import read_csv

from sklearn.ensemble import ExtraTreesClassifier

array = heart.values

X = array[:,0:13]

Y = array[:,13]

# feature extraction

model = ExtraTreesClassifier()

model.fit(X, Y)

print(model.feature_importances_)