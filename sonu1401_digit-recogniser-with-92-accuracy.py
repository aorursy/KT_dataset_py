import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import linear_model



from sklearn.model_selection import train_test_split



import gc

import cv2
digits = pd.read_csv('../input/train.csv')

print('The dimensions of digits dataset are - ', digits.shape)
import cv2

print(cv2.__version__)
print(digits.info())
digits.head()
four=digits.iloc[3,1:]



four.shape
three=digits.iloc[8,1:]

three.shape
four=four.values.reshape(28,28)
print(plt.imshow(four))
print(four[5:-5,5:-5])
#Summarize the counts of label to see how many digits are

digits.label.astype('category').value_counts()
#Summarize the counts in terms of percentage

round((digits.label.astype('category').value_counts()/digits.shape[0])*100,2)
digits.isnull().sum()
# Average values/distribution of features

description = digits.describe()

description
# Data Preparation for Model Building

X= digits.iloc[:,1:]

Y=digits.iloc[:,0]
# Rescaling the features

from sklearn.preprocessing import scale

X=scale(X)
# Train and Test size 

x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.10,random_state=101)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn import svm

from sklearn import metrics
# An Initial SVM Model with Linear Model

svm_linear= svm.SVC(kernel='linear')
svm_linear.fit(x_train,y_train)
# Prediction

predictions= svm_linear.predict(x_test)
predictions[:10]
# Evaluation : Confusion Matrix

metrics.confusion_matrix(y_test,predictions)
# Measure Accuracy

metrics.accuracy_score(y_test,predictions)
# Class-Wise Accuracy

class_wise=metrics.classification_report(y_test,predictions)
print(class_wise)
# to remove memory

gc.collect()
# rbf kernel with other parameters kept to default

svm_rbf = svm.SVC(kernel='rbf')

svm_rbf.fit(x_train,y_train)
# predict

predictions=svm_rbf.predict(x_test)
#accuracy

print(metrics.accuracy_score(y_test,predictions))
# conduct (grid_search) cross-validation to find the optimal values of cost C and the choice of kernel



from sklearn.model_selection import GridSearchCV



parameters = {'C':[1,10,100],

             'gamma':[1e-2,1e-3,1e-4]}



# Instantiate a model

svc_grid_search = svm.SVC(kernel='rbf')



# Create a classifier to perform grid search

clf= GridSearchCV(svc_grid_search,param_grid=parameters,scoring='accuracy')



# fit

clf.fit(x_train,y_train)
# results

cv_results = pd.DataFrame(clf.cv_results_)

cv_results         
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]
# plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

# plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

# plt.xlabel('C')

# plt.ylabel('Accuracy')

# plt.title("Gamma=0.01")

# plt.ylim([0.60, 1])

# plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

# plt.xscale('log')



# # subplot 2/3

# plt.subplot(132)

# gamma_001 = cv_results[cv_results['param_gamma']==0.001]



# plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

# plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

# plt.xlabel('C')

# plt.ylabel('Accuracy')

# plt.title("Gamma=0.001")

# plt.ylim([0.60, 1])

# plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

# plt.xscale('log')





# # subplot 3/3

# plt.subplot(133)

# gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



# plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

# plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

# plt.xlabel('C')

# plt.ylabel('Accuracy')

# plt.title("Gamma=0.0001")

# plt.ylim([0.60, 1])

# plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

# plt.xscale('log')



# plt.show()
# optimal hyperparameters

best_C = 1

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



print(test_accuracy, "\n")

print(confusion)
# class-wise accuracy

class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)

print(class_wise)