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
str_data = '../input/graduate-admissions/Admission_Predict_Ver1.1.csv'

df_data = pd.read_csv(str_data, engine='python')

df = df_data.drop('Serial No.', axis=1)

df.head()
# Normalize some columns 

X0 = df.iloc[:, :-1]

y = df.iloc[:, -1]



# Extend the X data 

one = np.ones((X0.shape[0], 1))

X = np.concatenate((X0, one), axis=1)

X_normalized = (X - X.min())/(X.max() - X.min())

X_normalized = X_normalized.reshape(500, 8)

y = np.array(y).reshape(500, 1)
def h_theta(X, theta): 

    h_ = X.dot(theta)

    return h_
def sgn(z): 

    h_ = 1.0/(1 + np.exp(z))

    return h_
theta = np.random.normal(size=X.shape[1]).reshape(X.shape[1], 1)

h_theta_ = h_theta(X_normalized, theta)

sgn(h_theta_)
def loss_func(X, y, theta): 

    h_ = sgn(h_theta(X, theta))

    loss = np.sum(y * np.log(h_) + (1 - y) * np.log(1 - h_))

    return -loss / X.shape[0]
loss_func(X_normalized, y, theta)
# Implement Gradient Descent method for this datasets 

learning_rate = 0.001 

def compute_grad(X, y, theta): 

    return np.sum(X * (sgn(np.dot(X, theta)) - y))
compute_grad(X_normalized, y, theta)
theta_ = theta 

for i in range(10): 

    theta_ = theta_ - learning_rate * compute_grad(X, y, theta_)

    print(loss_func(X_normalized, y, theta_))