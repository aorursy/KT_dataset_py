import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

import statsmodels.api as sm

sns.set()
df = pd.read_csv('../input/101-simple-linear-regressioncsv/1.01. Simple linear regression.csv')

df.head(3)
df.info()
df.describe()
missingno.matrix(df)
df.shape
y = df['GPA']

x1 = df[['SAT']]
plt.scatter(y,x1)

plt.xlabel('GPA', fontsize=20)

plt.ylabel('SAT', fontsize=20)

plt.show()
x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()
new_data = pd.DataFrame({'conts':1, 'SAT': [1900,1987,1690]})

new_data 
predictions = results.predict(new_data)

predictions
pred_df = pd.DataFrame({'predictions': predictions})

joined = new_data.join(pred_df)

joined