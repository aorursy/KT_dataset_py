# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df= pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
df.head(3)

data = df.values
data
X= data[:,1:]
Y= data[:,0]
print(X.shape,Y.shape)
image = X[4].reshape((28,28))
print(Y[4])
plt.imshow(image, cmap='pink')
plt.show()
#dividing data
split= int(0.70*X.shape[0])
print(split)
X_train,Y_train= X[:split,:],Y[:split]
print(X_train.shape,Y_train.shape)

X_test,Y_test=X[split:,:],Y[split:]









