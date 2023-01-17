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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

df.head()
df.describe()
x = df[['item_id','item_category_id']]

x.head()
viz = x[['item_id','item_category_id']]

viz.hist()

plt.show()
plt.scatter(x.item_id , x.item_category_id , color = 'red')

plt.xlabel('item_id')

plt.ylabel('item_category_id')

plt.show()
msk = np.random.rand(len(x)) >0.8

train = x[msk]

test = x[~msk]
from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['item_id']])

train_y = np.asanyarray(train[['item_category_id']])

regr.fit(train_x,train_y)

print('Coeffient',regr.coef_)

print('intercept',regr.intercept_)
plt.scatter(train.item_id,train.item_category_id,color = 'green',)

plt.plot(train_x,regr.coef_[0][0]*train_x + regr.intercept_[0],'r')

plt.xlabel('item_id')

plt.ylabel('item_category_id')

plt.grid("true")

plt.show()
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['item_id']])

test_y = np.asanyarray(test[['item_category_id']])

test_y = regr.predict(test_x)

print('mean_absolute_error:.2f'%np.mean(np.absolute(test_y - test_y)))

print('r2_score:%2f'%r2_score(test_y,test_y))