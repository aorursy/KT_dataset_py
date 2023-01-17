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
import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

raw_data = pd.read_csv('../input/1.03. Dummies.csv')
raw_data
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes':1,'No':0})
data
data.describe()
y = data['GPA']

x1 = data[['SAT','Attendance']]

x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()
plt.scatter(data['SAT'],y,c=data['Attendance'],cmap='RdYlGn_r')

yhat_no = 0.6439 + 0.0014*data['SAT']

yhat_yes = 0.8665 + 0.0014*data['SAT']

yhat = 0.0017*data['SAT'] + 0.275

fig = plt.plot(data['SAT'],yhat_no, lw=2, C='#006837', label ='regression line1')

fig = plt.plot(data['SAT'],yhat_yes, lw=2, C='#a50026', label ='regression line2')

fig = plt.plot(data['SAT'],yhat, lw=3, c='#4C72B0', label ='regression line')

plt.xlabel('SAT', fontsize = 20)

plt.ylabel('GPA', fontsize = 20)

plt.show()