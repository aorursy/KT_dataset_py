# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt # visualization
%matplotlib inline
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv').dropna()
x, y = np.array(df['x']), np.array(df['y'])
A = np.array([[sum(x**2), sum(x)],
              [   sum(x), len(x)]])
b = np.array([sum(x*y), sum(y)])
c = np.linalg.solve(A, b)
print('y = %.3f*x %+.3f' % (c[0], c[1]))

x_plt = np.linspace(min(x), max(x), 2)
plt.plot(x, y, '.')
plt.plot(x_plt, c[0]*x_plt + c[1])
plt.legend(['known points', 'fitted line'])
plt.title('Train file')
df = pd.read_csv('../input/test.csv').dropna()
x, y = np.array(df['x']), np.array(df['y'])
A = np.array([[sum(x**2), sum(x)],
              [   sum(x), len(x)]])
b = np.array([sum(x*y), sum(y)])
c = np.linalg.solve(A, b)
print('y = %.3f*x %+.3f' % (c[0], c[1]))

x_plt = np.linspace(min(x), max(x), 2)
plt.plot(x, y, '.')
plt.plot(x_plt, c[0]*x_plt + c[1])
plt.legend(['known points', 'fitted line'])
plt.title('Test file')