# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# libraries

import matplotlib.pyplot as plt

import seaborn as sns
# Reading the dataset

data_train = pd.read_csv("/kaggle/input/letterrecognition-using-svm/letter-recognition.csv")

data_train.head()
data_train.info()
# dimension of dataset

data_train.shape
# columns of dataset

print(data_train.columns)
# printing the letter in correct sequence

sequence = list(np.sort(data_train['letter'].unique()))

print(sequence)
# getting mean of columns for each alphabet

data_train_mean = data_train.groupby('letter').mean()

data_train_mean.head()
X = data_train.drop(['letter'],axis = 1)

y = data_train['letter']
# Scaling

from sklearn.preprocessing import scale

X = scale(X)
from sklearn.model_selection import train_test_split



# Spliting the dataset into train-test

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=101)
from sklearn.svm import SVC



# Building a linear SVM model

linear_model = SVC(kernel='linear')

linear_model.fit(X_train,y_train)



y_pred = linear_model.predict(X_test)
from sklearn import metrics

from sklearn.metrics import confusion_matrix



# accuracy and confusion matrix

print("accuracy :", metrics.accuracy_score(y_true = y_test,y_pred = y_pred), "\n")

print("confusion_matrix :", metrics.confusion_matrix(y_true = y_test,y_pred = y_pred))
#Building a non-linear SVM model

non_linear_model = SVC(kernel = 'rbf')

non_linear_model.fit(X_train,y_train)



# Predict

y_pred = non_linear_model.predict(X_test)
# accuracy and confusion matrix for non-linear SVM model

print("accuracy :", metrics.accuracy_score(y_true = y_test,y_pred = y_pred), "\n")

print("confusion_matrix :", metrics.confusion_matrix(y_true = y_test,y_pred = y_pred))
# hypertuning

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 101)



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

plt.ylim([0.60, 1])

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

plt.ylim([0.60, 1])

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

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')

# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# Building and Evaluating a final model

# model with optimal hyperparameters



# model

model = SVC(C=1000, gamma=0.01, kernel="rbf")



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

print(metrics.confusion_matrix(y_test, y_pred), "\n")