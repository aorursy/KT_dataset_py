# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline 
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/weight-height.csv')
df.head()
df.plot(kind = 'scatter',
               x = 'Height',
               y = 'Weight',
               title = 'Weight and Height in adults')
df.plot(kind = 'scatter',
               x = 'Height',
               y = 'Weight',
               title = 'Weight and Height in adults')
plt.plot([55,78],[75,250],color = 'red',linewidth = 3)
def line(x, w=0 , b=0):
    return x*w + b
x = np.linspace(55,80,100)
yhat = line(x ,w=0, b =0)

df.plot(kind = 'scatter',
               x = 'Height',
               y = 'Weight',
               title = 'Weight and Height in adults')
plt.plot(x,yhat,color = 'red',linewidth = 3)
def mean_squared_error(y_true, y_pred):
    s = (y_true-y_pred)**2
    return s.mean()
X = df[['Height']].values
y_true = df['Weight'].values
y_pred = line(X)
mean_squared_error(y_true,y_pred)