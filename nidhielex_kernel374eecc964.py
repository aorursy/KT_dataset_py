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
# Import important libraries

import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import scale
# import train.csv and test.csv

Train = pd.read_csv('../input/train.csv')

Test = pd.read_csv("../input/test.csv")

# data visualization

Train.head()
Test.head()
Train.tail()
# data types

Train.info()
# data types

Test.info()
# dimensions

print("Dimensions: ", Train.shape, "\n")
# dimensions

print("Dimensions: ", Test.shape, "\n")
Train.describe
Test.describe
# a quirky bug: the column names have a space, e.g. 'xbox ', which throws and error when indexed

print(Train.columns)
# a quirky bug: the column names have a space, e.g. 'xbox ', which throws and error when indexed

print(Test.columns)
Train.isnull().sum()
Test.isnull().sum()
Train.isnull().values.any()
Test.isnull().values.any()
# look at fraction

Train['label'].describe()
Train.describe()
Test.describe()
order = list(np.sort(Train['label'].unique()))

print(order)
Train_means = Train.groupby('label').mean()

Train_means.head()
pd.set_option('display.max_columns', 785)

Train.describe()
plt.figure(figsize=(30, 24))

sns.heatmap(Train_means)
#See the distribution of the labels

sns.countplot(Train.label)
Train.label.value_counts(dropna = False)

Number = Train[0:8000]



y = Number.iloc[:,0]



X = Number.iloc[:,1:]



print(y.shape)

print(X.shape)
# train test split

X_scaled = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)
# linear model



model_linear = SVC(kernel='linear')

model_linear.fit(X_train, y_train)



# predict

y_pred = model_linear.predict(X_test)
# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
# Accuracy

classification_metrics = metrics.classification_report(y_true=y_test, y_pred=y_pred)

print(classification_metrics)
# non-linear model

# using rbf kernel, C=1, default value of gamma



# model

non_linear_model = SVC(kernel='rbf')



# fit

non_linear_model.fit(X_train, y_train)



# predict

y_pred = non_linear_model.predict(X_test)
# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
# Accuracy

classification_metrics = metrics.classification_report(y_true=y_test, y_pred=y_pred)

print(classification_metrics)
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 0)



# specify range of hyperparameters

# Set the parameters by cross-validation

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]}]





# specify model

model = SVC(kernel="rbf")



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = model, 

                        param_grid = hyper_params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv.fit(X_train, y_train)                  

# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.70, 1.20])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 2/3

plt.subplot(132)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.80, 1.20])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')





# subplot 3/3

plt.subplot(133)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

plt.ylim([0.80, 1.20])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')

# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# model with optimal hyperparameters



# model

model = SVC(C=10, gamma=0.001, kernel="rbf")



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

print(metrics.confusion_matrix(y_test, y_pred), "\n")