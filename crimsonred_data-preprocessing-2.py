# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from sklearn.datasets import load_iris

from sklearn import datasets, linear_model, metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = load_iris()

a = dataset.data

b = np.zeros(150)
for i in range (150):

    b[i]=a[i,1]
b=np.sort(b)
bin1=np.zeros((30,5))

bin2=np.zeros((30,5))

bin3=np.zeros((30,5))
for i in range (0,150,5):

    k=int(i/5)

    mean=(b[i] + b[i+1] + b[i+2] + b[i+3] + b[i+4])/5

    for j in range(5):

        bin1[k,j]=mean

print("Bin Mean: \n",bin1)
for i in range (0,150,5):

    k=int(i/5)

    for j in range (5):

        if (b[i+j]-b[i]) < (b[i+4]-b[i+j]):

            bin2[k,j]=b[i]

        else:

            bin2[k,j]=b[i+4]

print("Bin Boundaries: \n",bin2)
for i in range (0,150,5):

    k=int(i/5)

    for j in range (5):

        bin3[k,j]=b[i+2]

print("Bin Median: \n",bin3)
#2

df_ages = pd.DataFrame({'age': np.random.randint(21, 51, 8)})

df_ages
from numpy import mean

from numpy import std

from numpy.random import randn

from numpy.random import seed

from numpy import cov

from scipy.stats import pearsonr

from matplotlib import pyplot
# seed random number generator

seed(1)

# prepare data

data1 = 20 * randn(1000) + 100

data2 = data1 + (10 * randn(1000) + 50)

# summarize

print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))

print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))

# plot

pyplot.scatter(data1, data2)



pyplot.show()
# calculate covariance matrix

covariance = cov(data1, data2)

print(covariance)
# calculate Pearson&#39;s correlation

corr, _ = pearsonr(data1, data2)

print('Pearsons correlation: %.3f' % corr)
df_ages['age_bins'] = pd.cut(x=df_ages['age'], bins=[20,29,30,39,40,49])

df_ages

df_ages['age_bins'].unique()

df_ages['age_by_decade'] = pd.cut(x=df_ages['age'], bins=[20, 29,39,49], labels=['20s', '30s', '40s'])

df_ages