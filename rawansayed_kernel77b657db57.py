# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Data Preprocessing



#Importing the libraries

import numpy as np 

import matplotlib.pyplot as plt 

import pandas as pd



#importing dataset

Yy= pd.read_csv('../input/machathon-10-filteration-test/sample_submission.csv')

train_set= pd.read_csv('../input/machathon-10-filteration-test/train.csv')

test_set= pd.read_csv('../input/machathon-10-filteration-test/test.csv')

y_test=Yy.iloc[: , 1].values

X_train = train_set.iloc[: , [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,39]].values

Y_train = train_set.iloc[: , 38].values



#feature scaling

from sklearn.preprocessing import StandardScaler

sc_X= StandardScaler()

X_train=sc_X.fit_transform(X_train)

Y_test=sc_X.fit_transform(test_set)



#knn method

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=95, metric='minkowski',p=2)

classifier.fit(X_train,Y_train)

y_pred=classifier.predict(Y_test)



from sklearn.metrics import confusion_matrix

cm =confusion_matrix(y_test,y_pred)

accuracy=np.mean(y_pred == y_test)





# Fitting SVR to the dataset

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train,Y_train)

# Predicting a new result

y_test = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,39]])    # Input to .predict must be 2-dimensional

# Visualising the SVR results

plt.scatter(X_train,Y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Truth or Bluff (SVR)')

plt.xlabel('WellID')

plt.ylabel('TotalOilInNext6Months')

plt.show()



# Visualising the SVR results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (SVR)')

plt.xlabel('WellID')

plt.ylabel('TotalOilInNext6Months')

plt.show()