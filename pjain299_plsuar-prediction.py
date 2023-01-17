# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split #test train split

import matplotlib.pyplot as plt# basic plotting library

import seaborn as sns# more advanced visual plotting library

from sklearn import preprocessing #preposscessing of data

from sklearn.ensemble import RandomForestRegressor #random forest algorithm

from sklearn.pipeline import make_pipeline #making pipeline for using multiple fuction simultaniously

from sklearn.model_selection import GridSearchCV #Crossvalidation

from sklearn.metrics import mean_squared_error, r2_score #calculating error

from sklearn.externals import joblib #to save model



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")
data.shape
data.head()
data.info()
data.isnull().sum().sum()
data.describe()
data.corr()
sns.pairplot(data=data,

             palette="husl",

             hue="target_class",

             vars=[" Mean of the integrated profile",

                   " Excess kurtosis of the integrated profile",

                   " Skewness of the integrated profile",

                   " Mean of the DM-SNR curve",

                   " Excess kurtosis of the DM-SNR curve",

                   " Skewness of the DM-SNR curve"])



plt.suptitle("PairPlot of Data Without Std. Dev. Fields",fontsize=18)



plt.tight_layout()

plt.show()   # pairplot without standard deviaton fields of data
plt.figure(figsize=(16,10))



plt.subplot(2,2,1)

sns.violinplot(data=data,y=" Mean of the integrated profile",x="target_class")



plt.subplot(2,2,2)

sns.violinplot(data=data,y=" Mean of the DM-SNR curve",x="target_class")



plt.subplot(2,2,3)

sns.violinplot(data=data,y=" Standard deviation of the integrated profile",x="target_class")



plt.subplot(2,2,4)

sns.violinplot(data=data,y=" Standard deviation of the DM-SNR curve",x="target_class")





plt.suptitle("ViolinPlot",fontsize=20)



plt.show()
y = data.target_class

x = data.drop('target_class', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, 

                                                    test_size=0.2, 

                                                    random_state=299, 

                                                    stratify=y)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
print(x_train_scaled.mean(axis=0))

print(x_train_scaled.std(axis=0))
pipeline = make_pipeline(preprocessing.StandardScaler(), 

                         RandomForestRegressor(n_estimators=100))
print(pipeline.get_params())
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],

                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

 

# Fit and tune model

clf.fit(x_train, y_train)
print(clf.refit)

print(clf.best_params_)
y_pred = clf.predict(x_test)
print(r2_score(y_test, y_pred))



print(mean_squared_error(y_test, y_pred))
#joblib.dump(clf, 'rf_regressor.pkl')

#clf2 = joblib.load('rf_regressor.pkl')

 

# Predict data set using loaded model #

#clf2.predict(X_test)