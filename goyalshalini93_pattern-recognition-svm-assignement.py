import warnings

warnings.filterwarnings('ignore')



#importing libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn import metrics
# Reading the dataset

dig_df = pd.read_csv("../input/train.csv")

dig_df.head()
dig_df.info()
dig_df.describe()
# Checking of any values are missing

dig_df.isnull().sum().sort_values(ascending = False)
# Counting the number of labels present in the dataset

dig_df.label.astype('category').value_counts()
100*(round(dig_df.label.astype('category').value_counts()/len(dig_df.index), 4))
sns.countplot(x = 'label', palette="Set3", data = dig_df)
# Splitting the data into train and test set

X = dig_df.iloc[:, 1:]

Y = dig_df.iloc[:, 0]



# Rescaling the features

from sklearn.preprocessing import scale

X = scale(X) 



# train test split with train_size=10% and test size=90%

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.20, random_state=101)
#building model

linear_model = svm.SVC(kernel='linear')



#fitting the model

linear_model.fit(x_train, y_train)
# Prediction on y

y_pred = linear_model.predict(x_test)
# Model evaluation (Using Accuracy score)

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
# Confusion Matrix

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
# rbf kernel with other hyperparameters kept to default 

rbf_model = svm.SVC(kernel='rbf')

rbf_model.fit(x_train, y_train)



# Prediction

y_pred = rbf_model.predict(x_test)



# Evaluation

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
# creating a KFold object with 5 splits 

from sklearn.model_selection import KFold

folds = KFold(n_splits = 5, shuffle = True, random_state = 101)



from sklearn.model_selection import GridSearchCV



parameters = {'C':[1, 10, 100], 

             'gamma': [1e-2, 1e-3, 1e-4]}



# instantiate a model 

model = svm.SVC(kernel="rbf")



# create a classifier to perform grid search

model_cv = GridSearchCV(model, param_grid=parameters, scoring='accuracy',cv= folds,verbose = 1, return_train_score=True)



# fit

model_cv.fit(x_train, y_train)
# results

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

plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

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

plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

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

plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

plt.xscale('log')



plt.show()
# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# optimal hyperparameters

best_C = 10

best_gamma = 0.001



# model

model_f = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)



# fit

model_f.fit(x_train, y_train)



# predict

y_pred= model_f.predict(x_test)
# confusion matrix and accuracy

print("accuracy", metrics.accuracy_score(y_true=y_test, y_pred= y_pred), "\n")

print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
