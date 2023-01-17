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

from sklearn.preprocessing import StandardScaler
train = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')

test = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')

submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')
sc = StandardScaler()
x_train = train.loc[:,'avgTemp':'rainFall']

y_train = train.loc[:,'avgPrice']

x_train = np.array(x_train)

y_train = np.array(y_train)
test = test.loc[:,'avgTemp':'rainFall']

test = np.array(test)
x_train = sc.fit_transform(x_train)

test = sc.fit_transform(test)
from sklearn.neighbors import KNeighborsRegressor



regressor = KNeighborsRegressor(n_neighbors = 500, weights = 'distance',p = 2)



regressor.fit(x_train,y_train)
pred = regressor.predict(test)

pred[:5]
for i in range(len(test)):

  submit['Expected'][i] = pred[i]
submit.to_csv('submission.csv',index=False,header=True)