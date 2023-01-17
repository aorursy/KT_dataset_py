# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading the dataset file

df = pd.read_csv('../input/HousePrices_HalfMil.csv', sep=';')
# Displaying the first 5 rows

df.head()
# Number of rows and columns in the DataFrame

df.shape
# Summary of information in all columns

df.describe().round(2)
# Correlation Matrix

df.corr().round(4)
#Investigating the dependent variable (y) according to a certain characteristic

ax = sns.boxplot(data = df['precos'], orient='v', width=0.2)

ax.figure.set_size_inches(12, 6)

ax.set_title('House Prices', fontsize=20)

ax.set_ylabel('Price', fontsize=16)

ax
ax = sns.boxplot(x = 'garagem', y = 'precos', data = df, orient='v', width=0.5)

ax.figure.set_size_inches(12, 6)

ax.set_title('Prices X Garagem', fontsize=16)

ax.set_ylabel('Price', fontsize=16)

ax.set_xlabel('Number of Garagem', fontsize=16)

ax
ax = sns.boxplot(x = 'banheiros', y = 'precos', data = df, orient='v', width=0.5)

ax.figure.set_size_inches(12, 6)

ax.set_title('Price X Banheiros', fontsize=20)

ax.set_ylabel('Price', fontsize=16)

ax.set_xlabel('Number of Banheiros', fontsize=16)

ax
ax = sns.boxplot(x = 'lareira', y = 'precos', data = df, orient='v', width=0.5)

ax.figure.set_size_inches(12, 6)

ax.set_title('Price X Lareira', fontsize=20)

ax.set_ylabel('Price', fontsize=16)

ax.set_xlabel('Number of Lareira', fontsize=16)

ax
ax = sns.boxplot(x = 'marmore', y = 'precos', data = df, orient='v', width=0.5)

ax.figure.set_size_inches(12, 6)

ax.set_title('Price X Acabamento em Mármore', fontsize=20)

ax.set_ylabel('Price', fontsize=16)

ax.set_xlabel('Acabamento em mármore branco (1) ou não (0)', fontsize=16)

ax
ax = sns.boxplot(x = 'andares', y = 'precos', data = df, orient='v', width=0.5)

ax.figure.set_size_inches(12, 6)

ax.set_title('Price X Andares', fontsize=20)

ax.set_ylabel('Price', fontsize=16)

ax.set_xlabel('Number of Andares', fontsize=16)

ax
ax = sns.pairplot(df, y_vars='precos', x_vars=['area', 'garagem', 'banheiros', 'lareira', 'marmore', 'andares'])

ax.fig.suptitle('Dispersion between variables', fontsize=20, y=1.10)

ax
ax = sns.pairplot(df, y_vars='precos', x_vars=['area', 'garagem', 'banheiros', 'lareira', 'marmore', 'andares'], kind='reg')

ax.fig.suptitle('Dispersion between variables', fontsize=20, y=1.10)

ax
from sklearn.model_selection import train_test_split
# Creating Series Dependent Variable

y = df['precos']
# Creating DataFrame Explanatory Variables

X = df[['area', 'garagem', 'banheiros', 'lareira', 'marmore', 'andares']]
# Creating datasets train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)
# Importing LinearRegression and metrics

from sklearn.linear_model import LinearRegression

from sklearn import metrics
model = LinearRegression()
model.fit(X_train, y_train)
print('R2 = {}'.format(model.score(X_train, y_train).round(2)))
y_predict = model.predict(X_test)
print('R2 = {}'.format(metrics.r2_score(y_test, y_predict).round(2)))
area = 38

garagem = 1

banheiros = 3

lareira = 1

marmore = 1

andares = 0

entrance = [[area, garagem, banheiros, lareira, marmore, andares]]



print('Predict price: {}'.format(model.predict(entrance)[0].round(2)))