# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read the brooklyn home sales CSV file
b_h_s = pd.read_csv('../input/brooklyn_sales_map.csv')
# Check the first few lines of transactions
b_h_s.head()
from sklearn.ensemble import RandomForestRegressor
train_y = b_h_s.sale_price
predictor_cols = ['block', 'lot', 'zip_code', 'residential_units', 'commercial_units', 'total_units', 'land_sqft', 'gross_sqft', 'year_built', 'year_of_sale']
train_X = b_h_s[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
predicted_prices = my_model.predict(train_X)
print(predicted_prices)
my_submission = pd.DataFrame({'Id': b_h_s.block, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('brooklyn_home_sales_prediction_submission.csv', index=False)