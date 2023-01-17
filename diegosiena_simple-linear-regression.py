# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



from matplotlib import pyplot as plt
df = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

df = df.dropna(0, 'any')
print('Data count %s' % df.count())

print('X Mean %s' % df['x'].mean())

print('Y Mean %s' % df['y'].mean())

print('X sum %s' % df['x'].sum())

print('Y sum %s' % df['y'].sum())

plt.scatter(df[['x']], df[['y']])

plt.show()
from sklearn import linear_model

modelo =  linear_model.LinearRegression()

modelo.fit(df[['x']], df[['y']])



print('(A) Intercepto: ', modelo.intercept_)

print('(B) Inclinação: ', modelo.coef_)
plt.scatter(df[['x']], df[['y']])

plt.plot(df[['x']], modelo.intercept_[0] + df[['x']]*modelo.coef_[0][0], '-r')

plt.show()
df_test = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')

y_predictions = modelo.predict(df_test[['x']])

plt.scatter(df_test[['x']], df_test[['y']], color="red")

plt.scatter(df_test[['x']], y_predictions)

plt.show()