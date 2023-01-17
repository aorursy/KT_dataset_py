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
import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.datasets import make_blobs

from sklearn.model_selection import GridSearchCV

import numpy as np, pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline



def showplot():

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)



    # plot the decision function

    ax = plt.gca()

    xlim = ax.get_xlim()

    ylim = ax.get_ylim()



    # create grid to evaluate model

    xx = np.linspace(xlim[0], xlim[1], 30)

    yy = np.linspace(ylim[0], ylim[1], 30)

    YY, XX = np.meshgrid(yy, xx)

    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = clf.decision_function(xy).reshape(XX.shape)



    # plot decision boundary and margins

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,

               linestyles=['--', '-', '--'])

    # plot support vectors

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,

               linewidth=1, facecolors='none', edgecolors='k')

    # plt.figure(figsize=(20,20))

    plt.show()
# we create 40 separable points

X, y = make_blobs(n_samples=40, centers=2, random_state=4)



# fit the model, don't regularize for illustration purposes

clf = svm.SVC(kernel='linear', C=10,gamma=1)

clf.fit(X, y)

showplot()

parameters = {'gamma': [0.01, 0.1, 10],

              'C': [0.001, 0.01, 0.1]}



model = svm.SVC(kernel = 'rbf')



grid = GridSearchCV(estimator = model, 

                    param_grid = parameters, 

                    cv = 5, 

                    n_jobs = -1, 

                    scoring = 'accuracy', 

                    verbose = 1, 

                    return_train_score = True)

grid.fit(X, y)
print('Score = %3.2f'%grid.score(X, y))

cv_results = pd.DataFrame(grid.cv_results_)

cv_results.head()



print('Best Score', grid.best_score_)

print('Best hyperparameters', grid.best_params_)
model_opt = svm.SVC(C = 0.001, gamma = 0.01, kernel = 'rbf')

model_opt.fit(X, y)
from sklearn import metrics

y_pred = model_opt.predict(X)

print('Accuracy', metrics.accuracy_score(y, y_pred))

print('Classification Report: \n', metrics.classification_report(y, y_pred))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Import the dataset using Seaborn library

iris=pd.read_csv('../input/iris/Iris.csv',index_col='Id')
iris.head()
# Creating a pairplot to visualize the similarities and especially difference between the species

sns.pairplot(data=iris, hue='Species', palette='Set2')
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
# Separating the independent variables from dependent variables

x=iris.iloc[:,:-1]

y=iris.iloc[:,4]

x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)
from sklearn.svm import SVC

model=SVC()

model.fit(x_train, y_train)
pred=model.predict(x_test)

print(confusion_matrix(y_test,pred))

print(classification_report(y_test, pred))
mnist_train = pd.read_csv('../input/mnist-dataset/mnist_train.csv', index_col = False)

mnist_test = pd.read_csv('../input/mnist-dataset/mnist_test.csv', index_col = False)
mnist_train.head()
def get_accuracy(X_train, X_validate, y_train, y_validate, k):

    '''fitting the SVC model for various kernels like linear, polynomial and rbf

    and finding the accuracy for each kernel for the train and validate sets. Based on the accuracy

    an appropriate kernel will be chosen for hyperparameter tuning and final model building with optimum hyperparameters'''

    

    #Caling teh scale_df() and storing the scaled df

    #X_train_scaled = scale_df(X_train)

    #X_validate_scaled = scale_df(X_validate)

    

    #Building a likear model for kernel type k passed in the parameter

    SVC_model = svm.SVC(kernel = k)

    

    #Fitting the model for the training set

    SVC_model.fit(X_train, y_train)

    #Predicting the labels for the training set

    y_train_predict = SVC_model.predict(X_train)

    #Accuracy for the training set

    train_accuracy = metrics.accuracy_score(y_train, y_train_predict)

    #Classification Report

    #c_report_train = metrics.classification_report(y_train, y_train_predict)

    

    #Fitting the model for validation set

    SVC_model.fit(X_validate, y_validate)

    #Predicting the labels for the validation set

    y_validate_predict = SVC_model.predict(X_validate)

    #Accuracy for the validation set

    validate_accuracy = metrics.accuracy_score(y_validate, y_validate_predict)

    #Classification Report

    #c_report_validate = metrics.classification_report(y_validate, y_validate_predict)

    

    #returning the accuracy for the train and validate set

    return(train_accuracy, validate_accuracy)
def scale_df(X):

    '''Scaling the data set using StandardScaler() and returning the scaled data set'''

    scale = StandardScaler()

    X_scaled = scale.fit_transform(X)

    return(X_scaled)



X_ = mnist_train.iloc[:, 1:]

y_ = mnist_train.iloc[:, 0]

print(X_.shape)

print(y_.shape)



X_train, X_validate, y_train, y_validate = train_test_split(scale_df(X_), y_, test_size = 0.20, random_state = 30, stratify = y_)

print(X_train.shape)

print(X_validate.shape)

print(y_train.shape)

print(y_validate.shape)
train_accuracy_linear, validate_accuracy_linear = get_accuracy(X_train, X_validate, y_train, y_validate, 'linear')

# train_accuracy_poly, validate_accuracy_poly = get_accuracy(X_train, X_validate, y_train, y_validate, 'poly')

# train_accuracy_rbf, validate_accuracy_rbf = get_accuracy(X_train, X_validate, y_train, y_validate, 'rbf')
print('Kernel = Linear')

print('Train Accuracy = ', train_accuracy_linear)

#print('Train Classification Report: \n', c_report_train_linear)

print('Validate Accuracy = ', validate_accuracy_linear)

#print('Validate Classification Report: \n', c_report_validate_linear)



# print('\n Kernel = Polynomial')

# print('Train Accuracy = ', train_accuracy_poly)

# #print('Train Classification Report: \n', c_report_train_poly)

# print('Validate Accuracy = ', validate_accuracy_poly)\

# #print('Validate Classification Report: \n', c_report_validate_poly)



# print('\n Kernel = RBF')

# print('Train Accuracy = ', train_accuracy_rbf)

# #print('Train Classification Report: \n', c_report_train_rbf)

# print('Validate Accuracy = ', validate_accuracy_rbf)

# #print('Validate Classification Report: \n', c_report_validate_rbf)