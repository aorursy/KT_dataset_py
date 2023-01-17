import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm
raw_data = pd.read_csv(r'/kaggle/input/Example-bank-data.csv')

raw_data.head()
data=raw_data.copy()
data= data.drop(['Unnamed: 0'],axis=1)

data.head()
data['y'] = data['y'].map({'yes':1, 'no':0})

data.head()
data.describe()
y=data['y']

x1=data['duration']
x= sm.add_constant(x1)

reg_log= sm.Logit(y,x)

results_log = reg_log.fit()

print(results_log.summary())
plt.scatter(x1,y,color = 'C1')

plt.xlabel('Duration', fontsize = 20)

plt.ylabel('Subscription', fontsize = 20)

plt.show()