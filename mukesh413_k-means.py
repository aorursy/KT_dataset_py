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
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# set three centers
center_1 = np.array([1,1])
center_2 = np.array([5,5])
center_3 = np.array([8,1])

# Genereting random data centering to the above three center

data_1 = np.random.randn(200, 2) + center_1
data_2 = np.random.randn(200, 2) + center_2
data_3 = np.random.randn(200, 2) + center_3

data = np.concatenate((data_1, data_2, data_3), axis =0)

plt.scatter(data[:, 0], data[:, 1], s = 2)
# Number of clusters
k = 3
# Number of traing examples
n = data.shape[0]
# Numbers of features in the dat
c = data.shape[1]

mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
centers = np.random.randn(k,c)*std + mean

# Plot the data and the data generated as random 
plt.scatter(data[:, 0], data[:, 1], s = 2)
plt.scatter(centers[:, 0], centers[:, 1], marker = '*', c = 'g', s = 150)
centers_old = np.zeros(centers.shape)
# to store old centers

#to store new centers
centers_new = deepcopy(centers)
data.shape
clusters = np.zeros(n)
destances = np.zeros((n, k))
error = np.linalg.norm(centers_new - centers_old)



