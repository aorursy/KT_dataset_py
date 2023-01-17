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
train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test_data.head()
from sklearn.linear_model import LinearRegression

features = ['YrSold' , 'PoolArea','LotArea','MSZoning']

x = pd.get_dummies(train_data[features])

y = train_data['SalePrice']

x_test  = pd.get_dummies(test_data[features])

model = LinearRegression()

model.fit(x,y)

Predictions = model.predict(x_test)
y
Predictions
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': Predictions})

output.to_csv('my.csv', index=False)

print("Your submission was successfully saved!")