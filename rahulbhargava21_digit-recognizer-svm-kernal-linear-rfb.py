import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import gc

import warnings

from IPython.display import Markdown, display ,HTML

from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns', 100)

warnings.filterwarnings('ignore')



def log(string):

    display(Markdown("> <span style='color:blue'>"+str(string)+"</span>"))
# read the dataset

digits = pd.read_csv("../input/train.csv")

digits.info()
# head

digits.head()
one = digits.iloc[2, 1:]

one.shape

one = one.values.reshape(28, 28)

plt.imshow(one, cmap='gray')
four = digits.iloc[3, 1:]

four.shape

four = four.values.reshape(28, 28)

plt.imshow(four, cmap='gray')
# visualise the array

print(four[5:-5, 5:-5])
# Summarise the counts of 'label' to see how many labels of each digit are present

count = pd.DataFrame(digits.label.astype('category').value_counts()).sort_index()

count = count.rename(columns={'label': 'Count'})
# Summarise count in terms of percentage 

percetage = pd.DataFrame(100*(round(digits.label.astype('category').value_counts()/len(digits.index), 4))).sort_index()

percetage = percetage.rename(columns={'label': 'Percetage'})

pd.concat([count, percetage], axis=1, join_axes=[count.index])
# missing values - there are none

#digits.isnull().sum()



## no null vales in dataset
# average values/distributions of features

description = digits.describe()

description
# Creating training and test sets

# Splitting the data into train and test

X = digits.iloc[:, 1:]

Y = digits.iloc[:, 0]



# Rescaling the features

from sklearn.preprocessing import scale

X = scale(X)



# train test split with train_size=10% and test size=90%

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.90, random_state=101)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)

# delete test set from memory, to avoid a memory error

# we'll anyway use CV to evaluate the model, and can use the separate test.csv file as well

# to evaluate the model finally



# del x_test

# del y_test
from sklearn import svm

from sklearn import metrics



# an initial SVM model with linear kernel   

svm_linear = svm.SVC(kernel='linear')



# fit

svm_linear.fit(x_train, y_train)
# predict

predictions = svm_linear.predict(x_test)

predictions[:10]
# evaluation: accuracy

# C(i, j) represents the number of points known to be in class i 

# but predicted to be in class j

confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)

confusion
# measure accuracy

log(metrics.accuracy_score(y_true=y_test, y_pred=predictions))
# class-wise accuracy

class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)

print(class_wise)
# run gc.collect() (garbage collect) to free up memory

# else, since the dataset is large and SVM is computationally heavy,

# it'll throw a memory error while training

log("Memory Claimed : "+str(gc.collect()))
# rbf kernel with other hyperparameters kept to default 

svm_rbf = svm.SVC(kernel='rbf')

svm_rbf.fit(x_train, y_train)
# predict

predictions = svm_rbf.predict(x_test)



# accuracy 

log(metrics.accuracy_score(y_true=y_test, y_pred=predictions))
# conduct (grid search) cross-validation to find the optimal values 

# of cost C and the choice of kernel







parameters = {'C':[1, 10, 100], 

             'gamma': [1e-2, 1e-3, 1e-4]}



# instantiate a model 

svc_grid_search = svm.SVC(kernel="rbf")



# create a classifier to perform grid search

clf = GridSearchCV(svc_grid_search, param_grid=parameters, scoring='accuracy',return_train_score=True)



# fit

clf.fit(x_train, y_train)
# results

cv_results = pd.DataFrame(clf.cv_results_)

cv_results
def plot_accuracy_graph(location,gamma_value) :

    plt.subplot(location)

    gamma = cv_results[cv_results['param_gamma']==gamma_value]

    plt.plot(gamma["param_C"], gamma["mean_test_score"])

    plt.plot(gamma["param_C"], gamma["mean_train_score"])

    plt.xlabel('C')

    plt.ylabel('Accuracy')

    plt.title("Gamma="+str(gamma_value))

    plt.ylim([0.60, 1])

    plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

    plt.xscale('log')
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plot_accuracy_graph(131,0.01)

plot_accuracy_graph(132,0.001)

plot_accuracy_graph(133,0.0001)



plt.show()
print(clf.best_score_)

print(clf.best_params_)
# optimal hyperparameters

best_C = 10

best_gamma = 0.001





# model

svm_final = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)



# fit

svm_final.fit(x_train, y_train)
# predict

predictions = svm_final.predict(x_test)
# evaluation: CM 

confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)



# measure accuracy

test_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)



log(test_accuracy)

print(confusion)
