# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

input_dir = "../input/"

# Any results you write to the current directory are saved as output.
X_train = np.load(input_dir + 'X_train.npy')

X_test = np.load(input_dir + 'X_test.npy')

y_train = np.load(input_dir + 'y_train.npy')

print("X_train shape :",X_train.shape)

print("X_test shape :",X_test.shape)

print("y_train shape :",y_train.shape)
pred = pd.DataFrame()

pred['Id'] = np.arange(X_test.shape[0])

pred['Prediction'] = np.zeros(X_test.shape[0],dtype = np.int)

pred.to_csv('pred.csv',index=False)

pred.head()