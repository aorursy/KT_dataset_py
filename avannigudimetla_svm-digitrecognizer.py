# Importing the libraries

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

from sklearn.metrics import classification_report

from sklearn.preprocessing import scale



# hide warnings

import warnings

warnings.filterwarnings('ignore')
import os

os.getcwd()
os.chdir('/kaggle')
os.chdir('input')
os.listdir()
df = pd.read_csv("train.csv")
df.head()
df.shape
df.info()
df.dtypes
round(df.isnull().sum()/len(df.index))
df.describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99,1])
# splitting into X and y

X = df.drop("label", axis = 1)

y = df['label']
#Scaling data

X_scaled = scale(X)
# train test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.8, random_state = 101)
# linear model



model_linear = SVC(kernel='linear')

model_linear.fit(X_train, y_train)



# predict

y_pred = model_linear.predict(X_test)

# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# confusion matrix

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
#printing the classification matrix

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



#Checking the precison and recall values

print("Precision Score :: ",metrics.precision_score(y_test,y_pred,pos_label='positive',average='micro'),"\n")



print("Recall Score :: ",metrics.recall_score(y_test,y_pred,pos_label='positive',average='micro'),"\n")
# non-linear model

# using rbf kernel, C=1, default value of gamma



# model

non_linear_model = SVC(kernel='rbf')



# fit

non_linear_model.fit(X_train, y_train)



# predict

y_pred = non_linear_model.predict(X_test)
#Checking the below values for the non-linear model



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))



#Checking the precison and recall values

print("Precision Score :: ",metrics.precision_score(y_test,y_pred,pos_label='positive',average='micro'),"\n")



print("Recall Score :: ",metrics.recall_score(y_test,y_pred,pos_label='positive',average='micro'),"\n")



#printing the classification matrix

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
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
#Choosing the best hyperparameters

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



print("Precision Score :: ",metrics.precision_score(y_test,y_pred,pos_label='positive',average='micro'))



print("Recall Score :: ",metrics.recall_score(y_test,y_pred,pos_label='positive',average='micro'))



from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



print(metrics.confusion_matrix(y_test, y_pred), "\n")