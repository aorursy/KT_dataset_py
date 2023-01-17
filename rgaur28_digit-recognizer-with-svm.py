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
# Importing libraries necessary for the study

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import validation_curve

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import svm

from sklearn import metrics
#Reading Train data

train = pd.DataFrame(pd.read_csv("/kaggle/input/digit-recognizer/train.csv"))

print(train.head())
#Reading test data

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(test.head())
train.shape
train.info()
train['label'].unique()
train.describe()
four=train.iloc[3,1:]

four.shape
four = four.values.reshape(28, 28)

plt.imshow(four, cmap='gray')
# Summarise count in terms of percentage 

100*(round(train.label.astype('category').value_counts()/len(train.index), 4))
#Let's now check the count of missing values 

train.isnull().sum()
# splitting into X and y

X = train.drop("label", axis = 1)

y = train.label.values.astype(int)
# Rescaling the features

from sklearn.preprocessing import scale

X = scale(X)
# split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.1, random_state = 4)
# an initial SVM model with linear kernel   

svm_linear = svm.SVC(kernel='linear')



# fit

svm_linear.fit(X_train, y_train)
#predict 

y_pred = svm_linear.predict(X_test)

y_pred[:10]
#confusion Matrix

metrics.confusion_matrix(y_test, y_pred)
#Accuracy Score

metrics.accuracy_score(y_test, y_pred)
#classification report

classification=metrics.classification_report(y_test, y_pred)

print(classification)
# SVM model with Non-linear kernel  

svm_rbf = svm.SVC(kernel='rbf')

svm_rbf.fit(X_train, y_train)
# predict

y_pred = svm_rbf.predict(X_test)



# accuracy 

print(metrics.accuracy_score(y_test, y_pred))
# Perform grid search CV to tune the hyperparameters C and gamma.

parameters = {'C':[1, 10, 100], 

             'gamma': [1e-2, 1e-3, 1e-4]}



# instantiate a model 

svc_grid_search = svm.SVC(kernel="rbf")



# create a classifier to perform grid search

clf = GridSearchCV(svc_grid_search, param_grid=parameters, scoring='accuracy')



# fit

clf.fit(X_train, y_train)
cv_results = pd.DataFrame(clf.cv_results_)

cv_results
clf.best_params_
best_C = 10

best_gamma = 0.001



# model

svm_final = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)



# fit

svm_final.fit(X_train, y_train)



# predict

y_pred = svm_final.predict(X_test)
X_test.shape
# accuracy

print("accuracy", metrics.accuracy_score(y_test, y_pred))



print("confusion matrix",metrics.confusion_matrix(y_test, y_pred))



# Rescaling the features

from sklearn.preprocessing import scale

test = scale(test)
prediction = svm_final.predict(test)
image_id = np.arange(1,prediction.shape[0]+1)

pd.DataFrame({"ImageId": image_id, "Label": prediction}).to_csv('svm_submission.csv', 

                                                                      index=False, header=True)