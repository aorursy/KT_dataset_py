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
train = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')

train.drop( columns='year',inplace=True)

train.head()
test = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')

test.drop( columns='year',inplace=True)

test.head()
submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')

submit.head()
dataset = train.values

X = dataset[:,0:4]

y = dataset[:,4]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1, random_state=1)



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)



from sklearn.neighbors import KNeighborsRegressor



regressor_test = KNeighborsRegressor(n_neighbors = 10, weights ='distance')



regressor_test.fit(X_train_std,y_train)
y_train_pred = regressor_test.predict(X_train_std)

y_test_pred = regressor_test.predict(X_test_std)

print('mean_error test samples: %d' %(y_test-y_test_pred).mean())
from sklearn.neighbors import KNeighborsRegressor



regressor = KNeighborsRegressor(n_neighbors = 9, weights ='distance')

X_std = sc.transform(X)

regressor.fit(X_std,y)
y_predict = regressor.predict(test.values)

y_predict
submit['Expected'] = y_predict

submit.head()
submit.to_csv('/kaggle/working/submit.csv',index=False)