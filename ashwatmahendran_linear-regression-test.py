# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import statsmodels.api as sm
import seaborn as sns
sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/random-linear-regression/test.csv')
data.describe()

# Any results you write to the current directory are saved as output.

y=data['y']
x1=data['x']
x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
results.summary()
plt.scatter(x1,y)
yhat=1.0143*x1+0.006
fig=plot(x1,yhat,lw=4,c='orange',label='regression line')
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
plt.show()
