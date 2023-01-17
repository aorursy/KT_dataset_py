# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import gc
digits = pd.read_csv('../input/digit-recognizer/train.csv')

digits.info()
digits.head()
digits.describe()
digits.isnull().sum()
Y=digits.label

Y.head()
X=digits.drop(['label'],axis=1)

X.head()
print(Y.shape)

print(X.shape)




# train test split with train_size=10% and test size=90%

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.70, random_state=101)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA as sklearnPCA



#standardized data

sc = StandardScaler().fit(x_train)

X_std_train = sc.transform(x_train)

X_std_test = sc.transform(x_test)



#If n_components is not set then all components are stored 

sklearn_pca = sklearnPCA().fit(X_std_train)

train_pca = sklearn_pca.transform(X_std_train)

test_pca = sklearn_pca.transform(X_std_test)



#Percentage of variance explained by each of the selected components.

#If n_components is not set then all components are stored and the sum of the ratios is equal to 1.0.

var_per = sklearn_pca.explained_variance_ratio_

cum_var_per = sklearn_pca.explained_variance_ratio_.cumsum()



plt.figure(figsize=(30,10))

ind = np.arange(len(var_per)) 

plt.bar(ind,var_per)

plt.xlabel('n_components')

plt.ylabel('Variance')
n_comp=len(cum_var_per[cum_var_per <= 0.90])

print("Keeping 90% Info with ",n_comp," components")

sklearn_pca = sklearnPCA(n_components=n_comp)

train_pca = sklearn_pca.fit_transform(X_std_train)

test_pca = sklearn_pca.transform(X_std_test)

print("Shape before PCA for Train: ",X_std_train.shape)

print("Shape after PCA for Train: ",train_pca.shape)

print("Shape before PCA for Test: ",X_std_test.shape)

print("Shape after PCA for Test: ",test_pca.shape)
from sklearn import svm

from sklearn import metrics
print("REKHA KAILAS-1")

svm_rbf = svm.SVC(kernel='rbf')

svm_rbf.fit(train_pca, y_train)

print("REKHA KAILAS-2")
print("REKHA KAILAS-3")

predictions = svm_rbf.predict(test_pca)



# accuracy 

print(metrics.accuracy_score(y_true=y_test, y_pred=predictions))
from sklearn.model_selection import GridSearchCV



parameters = {'C':[1, 10, 100], 

             'gamma': [1e-2, 1e-3, 1e-4]}



# instantiate a model 

svc_grid_search = svm.SVC(kernel="rbf")



# create a classifier to perform grid search

clf = GridSearchCV(svc_grid_search, cv=2,param_grid=parameters, scoring='accuracy')



# fit

clf.fit(train_pca, y_train)

print("REKHA KAILAS")
cv_results = pd.DataFrame(clf.cv_results_)

cv_results
best_score = clf.best_score_

best_hyperparams = clf.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
model = svm.SVC(C=10, gamma=0.001, kernel="rbf")

best_score = clf.best_score_

best_hyperparams = clf.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))



model.fit(train_pca, y_train)

y_pred = model.predict(test_pca)



# metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

print(metrics.confusion_matrix(y_test, y_pred), "\n")

digits_test = pd.read_csv('../input/digit-recognizer/test.csv')

digits_test.info()
X_std_test_actual = sc.transform(digits_test)

test_pca_actual = sklearn_pca.transform(X_std_test_actual)

print("Shape before PCA for Test: ",X_std_test_actual.shape)

print("Shape after PCA for Test: ",test_pca_actual.shape)
y_pred_actual = model.predict(test_pca_actual)
print(y_pred_actual)
print(type(y_pred_actual))
y_test_pred_df = pd.DataFrame(y_pred_actual)

y_test_pred_df.head()


len1 = len(y_pred_actual)

Type_new = list()

for i in range(len(y_pred_actual)): 

     Type_new.append(i+1)



y_test_pred_df.columns = ["Label"]

y_test_pred_df.insert(0,'ImageId',Type_new)

y_test_pred_df.head()





  
y_test_pred_df.to_csv('submission.csv',index=False) 