# libraries

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



import warnings

warnings.filterwarnings('ignore')
# dataset

train = pd.read_csv("../input/train.csv")

train.head()
mnist.shape
A, mnist=train_test_split(train, test_size = 0.40, random_state = 42)
mnist.shape
mnist.info()
mnist.describe
mnist.isnull().sum()
mnist.drop_duplicates(subset=None, keep='first', inplace=True)
mnist.shape
# lets see the distribution in numbers

mnist.label.astype('category').value_counts()
# splitting into X and y

X = mnist.drop("label", axis = 1)

y = mnist['label']
# scaling the features

X_scaled = scale(X)



# train test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)
# linear model



model_linear = SVC(kernel='linear')

model_linear.fit(X_train, y_train)



# predict

y_pred = model_linear.predict(X_test)

# confusion matrix



print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
# print other metrics



# accuracy

print("accuracy", metrics.accuracy_score(y_true=y_test, y_pred=y_pred),"\n")



# precision

print("precision", metrics.precision_score(y_true=y_test, y_pred=y_pred, average='macro'),"\n")



# recall/sensitivity

print("recall", metrics.recall_score(y_true=y_test, y_pred=y_pred, average='macro'),"\n")

# non-linear model

# using rbf kernel, C=1, default value of gamma



# model

non_linear_model = SVC(kernel='rbf')



# fit

non_linear_model.fit(X_train, y_train)



# predict

y_pred = non_linear_model.predict(X_test)
# confusion matrix



print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
# print other metrics



# accuracy

print("accuracy", metrics.accuracy_score(y_true=y_test, y_pred=y_pred),"\n")



# precision

print("precision", metrics.precision_score(y_true=y_test, y_pred=y_pred, average='macro'),"\n")



# recall/sensitivity

print("recall", metrics.recall_score(y_true=y_test, y_pred=y_pred, average='macro'),"\n")

# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)



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

cv_results.head()
# print 5 accuracies obtained from the 5 folds

print(cv_results)

print("mean accuracy = {}".format(cv_results.mean()))
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



test = pd.read_csv("../input/test.csv")

test1 = scale(test)
predicted_digit = model.predict(test1)
submission = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label': predicted_digit })
submission.to_csv("submission.csv",index=False)