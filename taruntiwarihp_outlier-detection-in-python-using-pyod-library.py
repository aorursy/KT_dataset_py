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
! pip install pyod
import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.font_manager
from pyod.models.abod import ABOD

from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers



#generate random data with two features

X_train, Y_train = generate_data(n_train=200,train_only=True, n_features=2)



# by default the outlier fraction is 0.1 in generate data function 

outlier_fraction = 0.1



# store outliers and inliers in different numpy arrays

x_outliers, x_inliers = get_outliers_inliers(X_train,Y_train)



n_inliers = len(x_inliers)

n_outliers = len(x_outliers)



#separate the two features and use it to plot the data 

F1 = X_train[:,[0]].reshape(-1,1)

F2 = X_train[:,[1]].reshape(-1,1)



# create a meshgrid 

xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))



# scatter plot 

plt.scatter(F1,F2)

plt.xlabel('F1')

plt.ylabel('F2') 
classifiers = {

     'Angle-based Outlier Detector (ABOD)'   : ABOD(contamination=outlier_fraction),

     'K Nearest Neighbors (KNN)' :  KNN(contamination=outlier_fraction)

}
#set the figure size

plt.figure(figsize=(10, 10))



for i, (clf_name,clf) in enumerate(classifiers.items()) :

    # fit the dataset to the model

    clf.fit(X_train)



    # predict raw anomaly score

    scores_pred = clf.decision_function(X_train)*-1



    # prediction of a datapoint category outlier or inlier

    y_pred = clf.predict(X_train)



    # no of errors in prediction

    n_errors = (y_pred != Y_train).sum()

    print('No of Errors : ',clf_name, n_errors)



    # rest of the code is to create the visualization



    # threshold value to consider a datapoint inlier or outlier

    threshold = stats.scoreatpercentile(scores_pred,100 *outlier_fraction)



    # decision function calculates the raw anomaly score for every point

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1

    Z = Z.reshape(xx.shape)



    subplot = plt.subplot(1, 2, i + 1)



    # fill blue colormap from minimum anomaly score to threshold value

    subplot.contourf(xx, yy, Z, levels = np.linspace(Z.min(), threshold, 10),cmap=plt.cm.Blues_r)



    # draw red contour line where anomaly score is equal to threshold

    a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')



    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score

    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')



    # scatter plot of inliers with white dots

    b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white',s=20, edgecolor='k') 

    # scatter plot of outliers with black dots

    c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black',s=20, edgecolor='k')

    subplot.axis('tight')



    subplot.legend(

        [a.collections[0], b, c],

        ['learned decision function', 'true inliers', 'true outliers'],

        prop=matplotlib.font_manager.FontProperties(size=10),

        loc='lower right')



    subplot.set_title(clf_name)

    subplot.set_xlim((-10, 10))

    subplot.set_ylim((-10, 10))

plt.show() 