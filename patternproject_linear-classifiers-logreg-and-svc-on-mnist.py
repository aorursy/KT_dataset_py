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
# Additional libraries

import matplotlib.pyplot as plt # for plots

%matplotlib inline

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold # for splits



from sklearn import metrics # for post-analysis
# reading in data files

data_train = pd.read_csv('../input/mnist_train.csv')

data_test = pd.read_csv('../input/mnist_test.csv')
print("The shape of TRAINING Data is: ", data_train.shape)

print("The shape of TEST Data is: ", data_test.shape)

# top rows

data_train.head(2)
data_test.head(2)
# What are the column names

print("Column Names")

data_train.columns
data_test.columns
# what are the data types of the columns?

print("Data Types of Column")

print(data_train.dtypes)

data_test.dtypes
data_train.describe() #seems useless
# checking for null values

data_train.isnull().values.sum()
data_test.isnull().values.sum()
# split the data set

X_train = np.array(data_train.iloc[:,1:])

y_train = np.array(data_train.iloc[:,0])
X_train.shape
y_train.shape
X_test = np.array(data_test.iloc[:,1:])

y_test = np.array(data_test.iloc[:,0])
X_test.shape
# sample rows

X_test[0:2,:]
y_test.shape
# sample rows

y_test[0:2]
from sklearn.svm import SVC
# takes way too long ...

# commenting out ...



#clf = SVC() 

#clf.fit(X_train,y_train)
#clf.score(X_train,y_train)
#y_predict = clf.predict(X_test)

#accuracy_score(y_predict,y_test)
from sklearn.preprocessing import StandardScaler



# Standardizing the features

scaler = StandardScaler()



# Fit on training set only.

scaler.fit(X_train)



# Apply transform to both the training set and the test set.

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_train_scaled[0:2,]
from sklearn.decomposition import PCA

def get_pca(x_train, x_test,pc_comp=0.95):

    pca = PCA(n_components=pc_comp)

    pca.fit(x_train)

    print("Explained Variance Ratios:", pca.explained_variance_ratio_)

    print("N Components are: ", pca.n_components_)

    x_train = pca.transform(x_train)

    x_test = pca.transform(x_test)

    print(x_train.shape, x_test.shape)

    return x_train, x_test
X_train_sd_pca, X_test_sd_pca = get_pca(X_train_scaled,X_test_scaled,0.5)
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults

# default solver is incredibly slow which is why it was changed to 'lbfgs'



clf_logReg = LogisticRegression(solver = 'lbfgs')
clf_logReg.fit(X_train_sd_pca, y_train)
# Predict for One Observation (image)

clf_logReg.predict(X_test_sd_pca[0].reshape(1,-1))
clf_logReg.score(X_test_sd_pca,y_test)
# picking these parameters at random to give SVM a spin first

clf_svm = SVC(kernel='linear', C=0.1)

clf_svm
clf_svm.fit(X_train_sd_pca, y_train)
y_pred_svm = clf_svm.predict(X_test_sd_pca)
print("SVM accuracy for TRAIN: ", clf_svm.score(X_train_sd_pca,y_train))

print("SVM accuracy for TEST: ", clf_svm.score(X_test_sd_pca,y_test))



print("LOG-REG accuracy for TRAIN: ", clf_logReg.score(X_train_sd_pca,y_train))

print("LOG-REG accuracy for TEST: ", clf_logReg.score(X_test_sd_pca,y_test))

# 1. Use GridSearchCV() to tune the parameter 'C' of linear SVM.



# create a parameter grid: map the parameter names to the values that should be searched

# simply a python dictionary

# key: parameter name

# value: list of values that should be searched for that parameter

# single key-value pair for param_grid



Cs = [0.001, 0.01, 0.1, 1, 10,100,500]

paramGrid = {'C':Cs}

#paramGrid = {'C':np.arange(0.01,500,10)}

#print(paramGrid)



# the cv with 5 does not finish, reducing it down to 2

#grid_search = GridSearchCV(clf_svm, paramGrid, cv=5)

grid_search = GridSearchCV(clf_svm, paramGrid, cv=2)
grid_search
# In previous iteration, even 0.5 PCAed data could not finish execution of Grid Search on Kaggle

# let us reduce it down to .25

# X_train_sd_pca2, X_test_sd_pca2 = get_pca(X_train_scaled,X_test_scaled,0.25)



# let us reduce it down to .15

# X_train_sd_pca2, X_test_sd_pca2 = get_pca(X_train_scaled,X_test_scaled,0.15)



# With Incremental Learning, trying again .25

X_train_sd_pca2, X_test_sd_pca2 = get_pca(X_train_scaled,X_test_scaled,0.25)





# need to combine scaled version of X_test and X_train

# X = np.vstack((X_test_sd_pca2,X_train_sd_pca2))

X = np.concatenate((X_test_sd_pca2,X_train_sd_pca2))



# need to combine original version of y_test and y_train

#y = np.vstack((y_test,y_train))



# diagnostic

print("the shapes of y are:", y_train.shape, y_test.shape)

print("the type of y are:", y_train.dtype, y_test.dtype)

print("the dim of y are:", y_train.ndim, y_test.ndim)



# np.vstack needs the length of two 1D arrays to be same, so switching to concatenate

y = np.concatenate((y_test,y_train))



# Grid Search 



#grid_search.fit(X, y)

#print("the best C value we get is: ",grid_search.best_params_)
# Grid Search - despite lowering variance retained in PCA and few value of C is too slow



# Alternate is to try "Incremental Learning"

# Details here :https://scikit-learn.org/0.15/modules/scaling_strategies.html



# new classifier - Stochastic Gradient with loss function hinge for SVC

from sklearn.linear_model import SGDClassifier

clf_sgd_svc = SGDClassifier()



clf_sgd_svc

# break train and test set into chunks



# diagnostic

print("the shape of X,y are:", X.shape, y.shape)

print("the type of X,y are:", X.dtype, y.dtype)

print("the dim of X,y are:", X.ndim, y.ndim)
# labels of all classes

class_labels = np.unique(y)

print(class_labels)
chunk_size = 50

X_size = X.shape[0]

iterations = (X_size / chunk_size)

#print(iterations)







for i in range(1,X_size,chunk_size):

    # outer loop 

    for j in range(i,i+chunk_size):

        # inner loop

        # add chunk row to the partial fit

        #print("second-loop")

        clf_sgd_svc.partial_fit(X[i:i+chunk_size,],y[i:i+chunk_size,], class_labels)

        #print(j)







print(clf_sgd_svc.score(X,y))
#help(range)
# try bagging and boosting

# https://www.kaggle.com/mayankkestwal10/ensemble-learning-bagging-and-boosting