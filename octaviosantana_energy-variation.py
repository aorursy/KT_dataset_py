"""
    Please ignore some typos, my english is not very good =)
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from datetime import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/glasses.csv', nrows=200000) ## GPU off
#df = pd.read_csv('../input/glasses.csv')         ## GPU on
df.head()
df['date'] = df['DATE'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f'))
df['seconds'] = df['date'].apply(lambda x: x-df['date'][0])
df['seconds'] = df['seconds'].apply(lambda x: x.value/10**9)
df.head()
df.shape
acc_x = np.array(df['ACC_X'].apply(lambda x: x**2))
acc_y = np.array(df['ACC_Y'].apply(lambda x: x**2))
acc_z = np.array(df['ACC_Z'].apply(lambda x: x**2))
time = np.array(df['seconds'])
## Reshape the variable
h=200
time = time.reshape((int(time.shape[0]/h),h))
acc_x = acc_x.reshape((int(acc_x.shape[0]/h),h))
acc_y = acc_y.reshape((int(acc_y.shape[0]/h),h))
acc_z = acc_z.reshape((int(acc_z.shape[0]/h),h))
"""
    Integration by the trapezium method
"""

dt = []
energy = []
t0 = 0
for i in range(h):
    t = float(time[-1,i] - time[0,i])#/10**9
    t0 += t    
    dt.append(t0)
    e = np.trapz(acc_x[:,i], x=time[:,i]) + np.trapz(acc_y[:,i], x=time[:,i]) + np.trapz(acc_z[:,i], x=time[:,i])
    energy.append(e)
    
energy = np.array(energy)
dt = np.array(dt)
energy_norm = energy/energy.max()
n0 = 100
nf = 145
plt.plot(dt[:n0], energy_norm[:n0], c='blue')
plt.plot(dt[n0:nf], energy_norm[n0:nf], c='red')
plt.plot(dt[nf:], energy_norm[nf:], c='blue')
plt.ylabel('$Energy$ $Expendire$ (arb. units)', fontsize=12)
plt.xlabel('$time(s)$', fontsize=12)
plt.show()
sns.distplot(energy_norm)
plt.show()
serie = pd.Series(energy_norm)

pd.plotting.lag_plot(serie, lag=1)
plt.xlabel('Energy (t)', fontsize=12)
plt.ylabel('Energy (t+1)', fontsize=12)
plt.show()
values = pd.DataFrame(serie.values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't']
result = dataframe.corr()
print(result)
from sklearn.ensemble import IsolationForest

n = 70
energy_dt = np.array(dataframe.iloc[1:])

model = IsolationForest()       ## Model
model.fit(energy_dt[:n])        ## Fit model
pred = model.predict(energy_dt) ## Predict
color = ['blue' if p==1 else 'red' for p in pred]

plt.scatter(energy_dt[:,0], energy_dt[:,1], c=color)
plt.xlabel('Energy (t)')
plt.ylabel('Energy (t+1)')
plt.show()
