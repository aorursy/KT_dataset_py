# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
from numpy.random import randn
dataset1 = randn(100)
dataset2 = randn(100)
plt.hist(dataset1)
plt.hist(dataset1,normed=True,color = 'red',alpha=.7,bins=20)
plt.hist(dataset2,normed=True,color = 'blue',bins=20,alpha=.6)
sns.jointplot(dataset1,dataset2)
sns.jointplot(dataset1,dataset2,kind='hex')
