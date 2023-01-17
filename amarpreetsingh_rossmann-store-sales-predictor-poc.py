import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# importing train data to learn
train = pd.read_csv("../input/train.csv", parse_dates = True, low_memory = False, index_col = 'Date')

# additional store data
store = pd.read_csv("../input/store.csv", low_memory = False)
train.head(5)
# adding new variable
train['SalePerCustomer'] = train['Sales']/train['Customers']
train['SalePerCustomer'].describe()
from statsmodels.distributions.empirical_distribution import ECDF
cdf = ECDF(train['Sales'])
train[ train['Store'] == 311 ]
