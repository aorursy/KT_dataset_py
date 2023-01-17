import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from scipy import stats

import statsmodels.api as sm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/autonobot-sp20-challenge-2/train_Level1Data.csv', sep=',')
data.head(20)
data.columns
plt.figure(figsize=(20,20), facecolor='white')

plt.scatter(data['X'], data['Y'])

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Scatter Plot for X & Y')

plt.tight_layout()

plt.grid(color='black', linestyle='dotted')

plt.show();
trainx = data.columns[0]

trainy = data.columns[1]

X = data[trainx]

Y = data[trainy]

X2 = sm.add_constant(X)

model = sm.OLS(Y, X2)

model_ = model.fit()

print(model_.summary())