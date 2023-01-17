# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

def formula(y, m):
    return m * y

slope = -510.25 #GUESS
X = [] #output

for y in range(1, 10):
    X.append(  formula(y, slope) )
X
#ABOVE BASICPYTHON

#BELOW is PANDAS
s = pd.Series(X)
s.plot.line()

#time series
data = np.random.randn(1000)

date_range=pd.date_range('1/1/2000', periods=1000)

ts = pd.Series(data, index=date_range)

#make it look like a trend instead of random brownian motion
ts = ts.cumsum()

ts.plot()


df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()

import matplotlib as plt
import matplotlib.pyplot as plt
plt.close('all')

plt.figure();
df.plot();