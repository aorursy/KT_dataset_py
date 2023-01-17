# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_palette('GnBu_d')

sns.set_style('whitegrid')



%matplotlib inline
startup = pd.read_csv('../input/sp-startup/50_Startups.csv') 

df = startup.copy()
df.head()
df.info()
df.shape
df.isna().sum().isna().sum()
df.corr()
sns.heatmap(df.corr());
sns.scatterplot(x='R&D Spend', y='Profit', data=df);
df.plot.hist(bins=10, alpha=0.5);
sns.pairplot(df);
df.describe().T
df['State'].unique()
dummy_states_df = pd.get_dummies(df['State'])

dummy_states_df.head()
df = pd.concat([df, dummy_states_df], axis=1)

df.drop(['Florida', 'State'], axis=1, inplace=True)

df.head()
X = df[['R&D Spend', 'Administration', 'Marketing Spend']]

y = df['Profit']
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
df_pred = pd.DataFrame({'actual': y_test, 'predictions': y_pred})

df_pred
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MSE = mean_squared_error(y_test, y_pred)

MAE = mean_absolute_error(y_test, y_pred)

print("MSE: ", MSE)

print("MAE: ", MAE)

print("RMSE: ", np.sqrt(MSE))
r2 = r2_score(y_test, y_pred)

print("R2 Score: ", r2)
import statsmodels.api as sm

sm_model = sm.OLS(y_train, X_train).fit()

sm_model.summary()