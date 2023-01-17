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
kddcup_data = pd.read_csv('../input/kddcup.data_10_percent_corrected.csv')
kddcup_data.head(5)
kddcup_data.dst_host_count.describe()
kddcup_data.dst_host_srv_coun.describe()
a = kddcup_data.dst_host_count.iloc[0]
b = kddcup_data.dst_host_srv_coun.iloc[0] 
x = np.array([(a,b)])
network = 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) 
    return np.exp(x) / np.sum(np.exp(x))

def predict(x):
    W1 = np.array((0.1,0.3,0.5),(0.2,0.4,0.6))
    W2 = np.array([0.1,0.3,0.5],[0.2,0.4,0.6])
    W3  = np.array([0.1,0.3,0.5],[0.2,0.4,0.6])
    b1 = np.array([0.1,0.2])
    b2 = np.array([0.1,0.2])
    b3 = np.array([0.1,0.2])

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

predict(x)