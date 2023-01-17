# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.optimize as opt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

X = train_data.iloc[:,1:].astype('float')

X_test = test_data.iloc[:,:].astype('float')

y = train_data.iloc[:,0]

m = len(X)

X_test.shape

# Any results you write to the current directory are saved as output.
ones = np.ones((m,1), dtype = 'float')

X = np.hstack((ones,X))

ones = np.ones((28000,1), dtype = 'float')

X_test = np.hstack((ones,X_test))

(a,b) = X.shape

X_test.shape


# sigmoid Function

def sigmoid(x):

    return 1/(1+np.exp(-x))
# cost function

def cost(theta, X, y):

        prediction = sigmoid(X @ theta)

        prediction[prediction == 1] = 0.99999 

        prediction[prediction == 0] = 0.00001

        #because log(1)=0 cause error in division

        J = (-1/m)*np.sum(np.multiply(y,np.log(prediction) 

                           +np.multiply(1-y,np.log(1- prediction))))

        return J



# Gradient funtion

def gradient(theta, X, y):

        return ((1/m) * (np.dot(X.T , (sigmoid(X @ theta) - y))))
# fitting the parameters

theta_optimized = np.zeros([10,b])

for i in range (0,10):

    label = (y == i).astype(int)

    initial_theta = np.zeros(X.shape[1])

    theta_optimized[i,:] = opt.fmin_cg(cost,initial_theta,gradient,(X,label))
# probability of each no 0 to 10 (28000*10)

tmp = sigmoid(X_test @ theta_optimized.T)

predictions = tmp.argmax(axis=1)

ID = list(range(1,28001))

my_submission = pd.DataFrame({'ImageId': ID, 'Label': predictions})

my_submission.to_csv('submission.csv', index=False)

my_submission.head()