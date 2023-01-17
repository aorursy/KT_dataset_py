import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor



test = pd.read_csv('../input/test.csv')

print(test.head())

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': test.YrSold})

my_submission.to_csv('mysubmission.csv', index=False)