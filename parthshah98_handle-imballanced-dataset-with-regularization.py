import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler, Normalizer

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

import warnings

from sklearn.feature_selection import RFE

warnings.filterwarnings("ignore")
def draw_line(coef,intercept, mi, ma):



    points=np.array([[((-coef[1]*mi - intercept)/coef[0]), mi],[((-coef[1]*ma - intercept)/coef[0]), ma]])

    plt.plot(points[:,0], points[:,1])
# here we are creating 2d imbalanced data points 

ratios = [(100,2), (100, 20), (100, 40), (100, 80)]

plt.figure(figsize=(20,5))

for j,i in enumerate(ratios):

    plt.subplot(1, 4, j+1)

    X_p=np.random.normal(0,0.05,size=(i[0],2))

    X_n=np.random.normal(0.13,0.02,size=(i[1],2))

    y_p=np.array([1]*i[0]).reshape(-1,1)

    y_n=np.array([0]*i[1]).reshape(-1,1)

    X=np.vstack((X_p,X_n))

    y=np.vstack((y_p,y_n))

    plt.scatter(X_p[:,0],X_p[:,1])

    plt.scatter(X_n[:,0],X_n[:,1],color='red')

plt.show()
C = [1000, 1, 0.01]

ratios = [(100,2), (100, 20), (100, 40), (100, 80)]



for c in C:

  plt.figure(figsize=(20,5))

  for j,i in enumerate(ratios):

      plt.subplot(1, 4, j+1)

      X_p=np.random.normal(0,0.05,size=(i[0],2))

      X_n=np.random.normal(0.13,0.02,size=(i[1],2))

      y_p=np.array([1]*i[0]).reshape(-1,1)

      y_n=np.array([0]*i[1]).reshape(-1,1)

      X=np.vstack((X_p,X_n))

      y=np.vstack((y_p,y_n))

      y = y.ravel() 



      #Creating a Model

      clf = LinearSVC(C=c)

      clf.fit(X,y)



      #plot_decision_regions(X=X, y=y,clf=clf, legend=2)

      

      plt.scatter(X_p[:,0],X_p[:,1])

      plt.scatter(X_n[:,0],X_n[:,1],color='red')

      draw_line(clf.coef_[0], clf.intercept_[0], -0.2,0.2)

  plt.show()
C = [1000, 1, 0.01]

ratios = [(100,2), (100, 20), (100, 40), (100, 80)]



for c in C:

  plt.figure(figsize=(20,5))

  for j,i in enumerate(ratios):

      plt.subplot(1, 4, j+1)

      X_p=np.random.normal(0,0.05,size=(i[0],2))

      X_n=np.random.normal(0.13,0.02,size=(i[1],2))

      y_p=np.array([1]*i[0]).reshape(-1,1)

      y_n=np.array([0]*i[1]).reshape(-1,1)

      X=np.vstack((X_p,X_n))

      y=np.vstack((y_p,y_n))

      y = y.ravel() 



      #Creating a Model

      clf = LogisticRegression(C=c)

      clf.fit(X,y)



      plt.scatter(X_p[:,0],X_p[:,1])

      plt.scatter(X_n[:,0],X_n[:,1],color='red')

      draw_line(clf.coef_[0], clf.intercept_[0], -0.2,0.2)

  plt.show()