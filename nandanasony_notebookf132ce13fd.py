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
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

train.head()



test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

test.head()
train_Y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd', ]



train_X = train[predictor_cols]



my_model = RandomForestRegressor()

my_model.fit(train_X, train_Y)





test_X = test[predictor_cols]

predicted_prices = my_model.predict(test_X)

print(predicted_prices)
submission_1 = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

submission_1.to_csv('submission1.csv', index=False)


