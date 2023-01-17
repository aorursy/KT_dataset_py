import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train.head()
features = ['LotArea','YearBuilt','FullBath','BedroomAbvGr','OverallQual']



X = train[features]

y = train['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



model = DecisionTreeRegressor(random_state=1)



model.fit(X_train,y_train)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test.head()
X_test = test[features]

y_pred = model.predict(X_test)



print(y_pred)
test_id = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')['Id']



submission = pd.DataFrame({'Id': test_id, 'SalePrice': y_pred})

submission.to_csv('lee_submission.csv', index=False)