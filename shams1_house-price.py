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
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor



# Read the data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



# pull data into target (y) and predictors (X)

train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']



# Create training predictors data

train_X = train[predictor_cols]



my_model = RandomForestRegressor()

my_model.fit(train_X, train_y)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test[predictor_cols]

# Use the model to make predictions

predicted_prices = my_model.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)