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
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df.head()

data=df.iloc[:,0:8]
data.head()
data.hist(figsize=(12,10))
plt.tight_layout()
plt.show()
colnames=data.columns.values
data.plot(kind='density', figsize=(12,10),subplots=True, layout=(3,3), sharex=False)
plt.tight_layout()
plt.show()
# Box and Whisker Plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,10))
plt.tight_layout()
plt.show()
#Correlation Matrix Plot
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
correlations = data.corr()
# plot correlation matrix
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
#Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(data,figsize=(16,16))
plt.show()
