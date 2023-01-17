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
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline   
plt.rcParams['image.cmap'] = 'gray'

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import argparse
import time
import cv2
def gen_mnist_image(X):
    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)
features = pd.read_csv('../input/mnist-data/train.csv').values[:,1:]
features = (features - np.min(features, 0)) / (np.max(features, 0) + 0.0001)  # 0-1 scaling
plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(features))
label = pd.read_csv('../input/mnist-data/train.csv').values[:,[0]]
# construct the training/testing split
from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(features, label, test_size = .3, random_state = 42)
trainX.shape, testX.shape
params = {"C": [1.0, 10.0, 100.0]}

start = time.time()

gs = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=5000), 
                  params, 
                  n_jobs = -1, 
                  verbose = 1)

gs.fit(trainX, trainY)
# print diagnostic information to the user and grab the best model
print ("done in %0.3fs" % (time.time() - start))
print ("best score: %0.3f" % (gs.best_score_))

bestParams = gs.best_estimator_.get_params()
bestParams
# initialize the RBM + Logistic Regression pipeline
rbm        = BernoulliRBM()
logistic   = LogisticRegression(max_iter=10000)

pipe       = Pipeline([("rbm", rbm), ("logistic", logistic)])
# rbm = BernoulliRBM(n_components = 200, 
#                    n_iter = 40,
#                    learning_rate = 0.01,  
#                    verbose = True)

# logistic = LogisticRegression(C = 1.0, max_iter=1000)

# # train the classifier and show an evaluation report
# classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
# classifier.fit(trainX, trainY)

# print (classification_report(testY, classifier.predict(testX)))
# initialize the RBM + Logistic Regression pipeline
rbm = BernoulliRBM()
logistic = LogisticRegression(max_iter=10000)
classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])

params = {
    "rbm__learning_rate": [0.1, 0.01],
    "rbm__n_iter": [50,100],
    "rbm__n_components": [100,200],
    "logistic__C": [1.0,5.0]}

# perform a grid search over the parameter
gsc = GridSearchCV(classifier, params, n_jobs = -1, verbose = 1, cv=2)
gsc.fit(trainX, trainY)

# print "Best Score: %0.3f" % (gsc.best_score_)

print ("RBM + Logistic Regression parameters",'\n')
bestParams = gsc.best_estimator_.get_params()
bestParams

# loop over the parameters and print each of them out
# so they can be manually set
# for p in sorted(params.keys()):
#     print "\t %s: %f" % (p, bestParams[p])
#Running Pipeline with best parameters as per GridSearchCV

rbm = BernoulliRBM(n_components = 200, 
                   n_iter = 20, #recommended is 100, reducing to complete in time
                   learning_rate = 0.1,  
                   verbose = True)

logistic = LogisticRegression(C = 1.0, max_iter=10000)

# train the classifier and show an evaluation report
classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
classifier.fit(trainX, trainY)

print (classification_report(testY, classifier.predict(testX)))
#Running Pipeline with best parameters as per GridSearchCV

rbm = BernoulliRBM(n_components = 200, 
                   n_iter = 100,
                   learning_rate = 0.1,  
                   verbose = True)

logistic = LogisticRegression(C = 1.0, max_iter=10000)

# train the classifier and show an evaluation report
classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
classifier.fit(trainX, trainY)

print (classification_report(testY, classifier.predict(testX)))