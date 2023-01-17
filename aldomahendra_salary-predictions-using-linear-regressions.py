import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm



sns.set()
df = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')

df.head()
df.isnull().sum()
x1 = df['YearsExperience']

y = df['Salary']
plt.scatter(x1,y)

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()
plt.scatter(x1,y)

yhat = 25790 + x1*9449.9623

fig = plt.plot(x1,yhat, lw=4, c='orange', label='Regression Line')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
new_data = pd.DataFrame({'conts': 1,'YearsExperience': [9, 12, 15, .5]})

predictions = results.predict(new_data)

predictions