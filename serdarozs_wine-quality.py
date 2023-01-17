# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

from IPython.display import Markdown as md



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from scipy.stats import skew

data = pd.read_csv("../input/winequality.csv")



# Get the number of the columns  

N = len(data.columns)

plt.figure(figsize=(8,6))



# Look at the distributions

i = 1 



for col in data.columns:

    plt.subplot(3,4,i)

    plt.title('Skewness: %f' % skew(data[col]))

    sns.distplot(data[col]);

    i += 1

plt.tight_layout();
data.head()
data.describe()
from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.gaussian_process import GaussianProcessClassifier



data_array = data.values.copy()

X,y = data_array[:,:11], data_array[:,11]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,random_state=10)



logistic_reg = linear_model.LogisticRegressionCV(cv=5,max_iter=10000)

logistic_reg.fit(X_train,y_train,)

logistic_reg.score(X_test,y_test)

print("Logistic regression score:", logistic_reg.score(X_test,y_test))



dt_cf = DecisionTreeClassifier()

dt_cf.fit(X_train,y_train)

print("Decision tree score:", dt_cf.score(X_test,y_test))



rf_cf = RandomForestClassifier(n_estimators=500)

rf_cf.fit(X_train,y_train)

print("Random forest score:", rf_cf.score(X_test,y_test))



gp_cf = GaussianProcessClassifier()

gp_cf.fit(X_train,y_train)

print("Gaussian process clasifier score:", gp_cf.score(X_test,y_test))



gb_cf = GradientBoostingClassifier()

gb_cf.fit(X_train,y_train)

print("Gradient boosting clasifier score:", gb_cf.score(X_test,y_test))



sgd_cf = linear_model.SGDClassifier()

sgd_cf.fit(X_train,y_train)

print("SGD classifier score:", sgd_cf.score(X_test,y_test))
import scikitplot as skplt



predicted_probas = rf_cf.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, predicted_probas,classes_to_plot=[3,4,5,6,7,8])

plt.legend(bbox_to_anchor=(1,1))
data_ = data.drop(['density'],axis=1)

data_array = data_.values.copy()

X,y = data_array[:,:10], data_array[:,10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,random_state=10)



logistic_reg = linear_model.LogisticRegressionCV(cv=5,max_iter=10000)

logistic_reg.fit(X_train,y_train,)

logistic_reg.score(X_test,y_test)

print("Logistic regression score:", logistic_reg.score(X_test,y_test))



dt_cf = DecisionTreeClassifier()

dt_cf.fit(X_train,y_train)

print("Decision tree score:", dt_cf.score(X_test,y_test))



rf_cf = RandomForestClassifier(n_estimators=500)

rf_cf.fit(X_train,y_train)

print("Random forest score:", rf_cf.score(X_test,y_test))



gp_cf = GaussianProcessClassifier()

gp_cf.fit(X_train,y_train)

print("Gaussian process clasifier score:", gp_cf.score(X_test,y_test))



gb_cf = GradientBoostingClassifier()

gb_cf.fit(X_train,y_train)

print("Gradient boosting clasifier score:", gb_cf.score(X_test,y_test))



sgd_cf = linear_model.SGDClassifier()

sgd_cf.fit(X_train,y_train)

print("SGD classifier score:", sgd_cf.score(X_test,y_test))
predicted_probas = rf_cf.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, predicted_probas,classes_to_plot=[3,4,5,6,7,8])

plt.legend(bbox_to_anchor=(1,1))
skplt.metrics.plot_confusion_matrix(y_test,rf_cf.predict(X_test))