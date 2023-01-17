# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.metrics as mt

import sklearn.linear_model as lr

import sklearn.preprocessing as pp

import sklearn.model_selection as ms

import matplotlib.pyplot as plt

import sklearn.preprocessing as pp





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/insurance/insurance.csv')

df.head()
le = pp.LabelEncoder()

df1 = df.select_dtypes(exclude=[np.number])

print(df1)
for col in df1.columns:

  df[col] = le.fit_transform(df[col])
print(df)

y = df.charges.values

df.drop('charges', axis=1, inplace=True)

x=df.values

print (x)
np.delete(x, 0, 1)

print(x)

print(y)

y.shape
print(x)
x.shape
x_train, x_test, y_train, y_test = ms.train_test_split(x,y, test_size=0.2, random_state=0)
reg = lr.LinearRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

score = mt.r2_score(y_test, y_pred)

print(score)
plt.scatter(x_test[:5, :1], y_pred[:5], color='r')

plt.scatter(x_test[:5, :1], y_test[:5], color='b')

plt.show()