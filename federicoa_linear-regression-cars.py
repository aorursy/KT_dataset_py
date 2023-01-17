import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model #linear model from sklearn

from sklearn.metrics import r2_score # evaluation

import seaborn as sns # plotting

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')

df.head()
y = df['mpg'].values

x = df['displacement'].values

print(x.shape)

print(y.shape)

print(x.ndim)

print(y.ndim)
x = x.reshape(-1, 1)

y = y.reshape(-1, 1)

print(x.shape)

print(y.shape)

print(x.ndim)

print(y.ndim)
sns.regplot(x=x, y=y)
# Create linear regression object

reg = linear_model.LinearRegression()

reg
# Train the model using the training sets

reg.fit(x,y)

# The coefficients

print(f'Coefficient: {reg.coef_[0][0]:.4f}') # we have only one coefficient, because we have only one feature

print(f'Intercept:   {reg.intercept_[0]:.4f}') # the intercept

print(f'Equation:    y = {reg.intercept_[0]:.4f} {reg.coef_[0][0]:.4f} x')
# create predicted values

y_hat = reg.predict(x)

y_hat[:10]
r2 = r2_score(y_hat , y)

print (f'R2 score: {r2:.2f}')
# plot

sns.despine()

fig, ax = plt.subplots(figsize=(12,12))

sns.scatterplot(x='displacement',y='mpg',data=df,s=100).set_title('Linear Regression\n MPG and Displacement', fontsize=20)

sns.lineplot(x=df['displacement'], y=y_hat[:,0], color='r', linewidth=10)

ax.set_xlim(0,max(x))

equation:str = f'Equation:    y = {reg.intercept_[0]:.4f} {reg.coef_[0][0]:.4f} x'

ax.text(200, 45,equation, fontsize=20) # equation text

ax.text(200, 42,f'R2 = {r2:.2f}',fontsize=20) # r2 score text