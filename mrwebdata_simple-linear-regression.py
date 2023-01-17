import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # seaborn for plots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from sklearn import linear_model
train = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

train.head()
sns.regplot(x=train.x, y=train.y, data=train)
regr = linear_model.LinearRegression()

regr
print(train['x'].shape)

print(train['y'].shape)
# check for nan values

print(sum(train['x'].isna()))

print(sum(train['y'].isna()))
train = train.dropna()
regr.fit(train[['x']],train.y) # use [['x']] instead of ['x'] because  we can only use predict on data that is of the same dimensionality as the training data (X) was. 
test = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')

test.head(5)
# evaluation

from sklearn.metrics import r2_score



test_y_hat = regr.predict(test[['x']])
print(f'Mean absolute error: {np.mean(np.absolute(test_y_hat - test.y)):.2f}')

print(f'Residual sum of squares (MSE): {np.mean((test_y_hat - test.y) ** 2):.2f}')

print(f'R2-score: {r2_score(test_y_hat , test.y):.2f}' )