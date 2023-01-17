# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# dataset

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info()
test.info()
# summing up the missing values (column-wise) in train data set

sum(round(100*(train.isnull().sum()/len(train.index)), 2)>0)
# summing up the missing values (column-wise) in test data set

sum(round(100*(train.isnull().sum()/len(train.index)), 2)>0)
train.nunique()==1
test.nunique()==1
## Visualizing the number of class and counts in the datasets

sns.countplot(train["label"])
# Plotting some samples

four = train.iloc[3, 1:]

four.shape

four = four.values.reshape(28,28)

plt.imshow(four, cmap='gray')

plt.title("Digit 4")
seven = train.iloc[6, 1:]

seven.shape

seven = seven.values.reshape(28, 28)

plt.imshow(seven, cmap='gray')

plt.title("Digit 7")
# basic plots: How do various pixels vary with the digits



plt.figure(figsize=(10, 5))

sns.barplot(x='label', y='pixel45', 

            data=train)
train_1 = train.drop('label',axis=1)
zero_val_cols_removal = pd.DataFrame(((train_1 != 0).any(axis=0) &  (test != 0).any(axis=0))==False)
zero_val_cols_removal.reset_index(inplace=True)

zero_val_cols_removal.columns = ['col_name','is_zero']
for col_name in zero_val_cols_removal.loc[(zero_val_cols_removal.is_zero==True),'col_name']:

    train.drop(col_name,axis=1,inplace=True)

    test.drop(col_name,axis=1,inplace=True)
train.info()
test.info()
train_sample = train.sample(frac =.20,random_state=10) 
train_sample.head()
train_sample.info()
X_train = train_sample.drop('label',axis=1)

y_train = train_sample['label']
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

test = scaler.transform(test)
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# specify range of parameters (C) as a list

params = {"C": [0.1, 1, 10, 100, 1000]}



model_linear = SVC(kernel='linear', cache_size=10000)



# set up grid search scheme

# note that we are still using the 5 fold CV scheme we set up earlier

model_cv = GridSearchCV(estimator = model_linear, 

                        param_grid = params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True,

                        n_jobs=-1)      
# fit the model - it will fit 5 folds across all values of C

model_cv.fit(X_train, y_train)  
# results of grid search CV

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plot of C versus train and test scores



plt.figure(figsize=(8, 6))

plt.plot(cv_results['param_C'], cv_results['mean_test_score'])

plt.plot(cv_results['param_C'], cv_results['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')
# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# model with optimal hyperparameters



# model

model = SVC(C=0.1,  kernel="linear")



model.fit(X_train, y_train)

y_pred = model.predict(test)
y_pred = pd.DataFrame(y_pred)
y_pred.head()
y_pred.reset_index(inplace=True)
y_pred.head()
y_pred.columns = ['ImageId','Label']
y_pred.ImageId = y_pred.ImageId + 1
y_pred.to_csv('result_linear.csv', index=False)
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# specify range of parameters (C) as a list

hyper_params = [ {'gamma': [1e-1, 1e-2],

                      'C': [0.1, 1],

                 'degree': [2,3]

                 }]





model_poly = SVC(kernel='poly', cache_size=10000)



# set up grid search scheme

# note that we are still using the 5 fold CV scheme we set up earlier

model_cv = GridSearchCV( estimator = model_poly, 

                         param_grid = hyper_params, 

                         scoring= 'accuracy', 

                         cv = folds, 

                         verbose = 1,

                         return_train_score=True,

                         n_jobs=-1)      
# fit the model - it will fit 5 folds across all values of C, gamma and degree

model_cv.fit(X_train, y_train) 
# results of grid search CV

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# # plotting

plt.figure(figsize=(25,20))



# subplot 4/1

plt.subplot(221)

gamma_1_degree_2 = cv_results.loc[(cv_results.param_gamma==0.1) & (cv_results.param_degree==2)]



plt.plot(gamma_1_degree_2["param_C"], gamma_1_degree_2["mean_test_score"])

plt.plot(gamma_1_degree_2["param_C"], gamma_1_degree_2["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.1 Degree=2")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 4/2

plt.subplot(222)

gamma_1_degree_3 = cv_results.loc[(cv_results.param_gamma==0.1) & (cv_results.param_degree==3)]



plt.plot(gamma_1_degree_3["param_C"], gamma_1_degree_3["mean_test_score"])

plt.plot(gamma_1_degree_3["param_C"], gamma_1_degree_3["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.1 Degree=3")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 4/3

plt.subplot(223)

gamma_01_degree_2 = cv_results.loc[(cv_results.param_gamma==0.01) & (cv_results.param_degree==2)]



plt.plot(gamma_01_degree_2["param_C"], gamma_01_degree_2["mean_test_score"])

plt.plot(gamma_01_degree_2["param_C"], gamma_01_degree_2["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01 Degree=2")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 4/4

plt.subplot(224)

gamma_01_degree_3 = cv_results.loc[(cv_results.param_gamma==0.01) & (cv_results.param_degree==3)]



plt.plot(gamma_01_degree_3["param_C"], gamma_01_degree_3["mean_test_score"])

plt.plot(gamma_01_degree_3["param_C"], gamma_01_degree_3["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01 Degree=3")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')

# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# model with optimal hyperparameters



# model

model = SVC(C=0.1, degree=3, gamma=0.1, kernel="poly")



model.fit(X_train, y_train)

y_pred = model.predict(test)
y_pred = pd.DataFrame(y_pred)
y_pred.head()
y_pred.reset_index(inplace=True)
y_pred.head()
y_pred.columns = ['ImageId','Label']
y_pred.ImageId = y_pred.ImageId + 1
y_pred.to_csv('result_polynomial.csv', index=False)
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# specify range of parameters (C) as a list

hyper_params = [ {'gamma': [1e-1, 1e-2, 1e-3],

                      'C': [0.1, 1, 10]

                 }]





model_poly = SVC(kernel='rbf', cache_size=10000)



# set up grid search scheme

# note that we are still using the 5 fold CV scheme we set up earlier

model_cv = GridSearchCV( estimator = model_poly, 

                         param_grid = hyper_params, 

                         scoring= 'accuracy', 

                         cv = folds, 

                         verbose = 1,

                         return_train_score=True,

                         n_jobs=-1)      
# fit the model - it will fit 5 folds across all values of C and gamma

model_cv.fit(X_train, y_train) 
# results of grid search CV

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# # plotting

plt.figure(figsize=(25,8))



# subplot 3/1

plt.subplot(131)

gamma_1 = cv_results.loc[(cv_results.param_gamma==0.1)]



plt.plot(gamma_1["param_C"], gamma_1["mean_test_score"])

plt.plot(gamma_1["param_C"], gamma_1["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.1")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 3/2

plt.subplot(132)

gamma_01 = cv_results.loc[(cv_results.param_gamma==0.01)]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 3/3

plt.subplot(133)

gamma_001 = cv_results.loc[(cv_results.param_gamma==0.001)]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.60, 1])

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

y_pred = model.predict(test)
y_pred = pd.DataFrame(y_pred)
y_pred.head()
y_pred.reset_index(inplace=True)
y_pred.head()
y_pred.columns = ['ImageId','Label']
y_pred.ImageId = y_pred.ImageId + 1
y_pred.to_csv('result_rbf.csv', index=False)