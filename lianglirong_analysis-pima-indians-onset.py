# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
names = ['preg','plas','pres','skin','mass','pedi','age','class']
data = pd.read_csv("../input/pima_data.csv",names=names)

data.head(10)
data.shape
data.dtypes
data.describe()
data.groupby('class').size()
data.corr()
data.skew()
#Visualizing Data
#histogram
data.hist()
plt.show()
#density graph
data.plot(kind="density",subplots=True,layout=(3,3),sharex=False)
plt.show()
#box graph
data.plot(kind="box",subplots=True,layout=(3,3),sharex=False)
plt.show()
#Correlation Matrix graph
correlation = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
#scatter matrix
pd.plotting.scatter_matrix(data,figsize=(10,10))
plt.show()
