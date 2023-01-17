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
# Making imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 9.0)





# Preprocessing Input data

data = pd.read_csv('/kaggle/input/dataset/data.csv')

X = data.iloc[:, 0]

Y = data.iloc[:, 1]

plt.scatter(X, Y)

plt.show()



# Building the model

X_mean = np.mean(X)

Y_mean = np.mean(Y)



num = 0

den = 0

for i in range(len(X)):

    num += (X[i] - X_mean)*(Y[i] - Y_mean)

    den += (X[i] - X_mean)**2

m = num / den

c = Y_mean - m*X_mean



print (m, c)



# Making predictions

Y_pred = m*X + c



plt.scatter(X, Y) # actual

plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted

plt.show()