# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from scipy import stats

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from datetime import datetime

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/gold-interest-rate-reg/regression test(1).csv')

df
df = df.iloc[:,1:]

df
df.head()
features = df[['gold_returns', 'interest_rate_pec_change']]

features.describe()
target = df['gold_returns']

split_num = int(len(features) * 0.8)

train_x = features[:split_num]

train_y = features[:split_num]

test_x = features[split_num:]

test_y = features[split_num:]
plt.title('interest_rate_pec_change vs gold_returns')

plt.ylabel('interest_rate_pec_change')

plt.xlabel('gold_returns')

plt.scatter(train_x, train_y)
print(train_x.shape, train_y.shape)
from sklearn.linear_model import LinearRegression 

model = LinearRegression()

model.fit(train_x, train_y)
model.coef_, model.intercept_
model.coef_
model.intercept_
preds = model.predict(test_x)

preds
def mae_value(y_true, y_pred):

    n = len(y_true)

    mae = sum(np.abs(y_true - y_pred))/n

    return mae

def mse_value(y_true, y_pred):

    n = len(y_true)

    mse = sum(np.square(y_true - y_pred))/n

    return mse
mae = mae_value(test_y.values, preds)

mse = mse_value(test_y.values, preds)



print("MAE: ", mae)

print("MSE: ", mse)
x = df['interest_rate_pec_change']

y = df['gold_returns']

plt.scatter(x, y)



plt.title('interest_rate_pec_change vs gold_returns')

plt.ylabel('interest_rate_pec_change')

plt.xlabel('gold_returns')
df.corr()
sns.regplot(x = 'interest_rate_pec_change', y = 'gold_returns', data = df)

plt.ylim(0,)



plt.ylabel('interest_rate_pec_change')

plt.xlabel('gold_returns')
x = df[['interest_rate_pec_change']]

y = df[['gold_returns']]

x, y
LM = LinearRegression()

LM.fit(x,y)
LM.intercept_
LM.coef_