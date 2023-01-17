import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

submit = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
print(train.describe())

print(train.head())
# 1. Combine the Country_Region and Province_State columns into country_province.

train.Province_State[train['Province_State'].isnull()] = ''

train['country_province'] = train.apply(lambda x: x.Country_Region if x.Province_State == '' else x.Country_Region+'_'+x.Province_State, axis = 1)

# train.country_province.unique()
# 2. Calculate the cumulative cases and fatalities for each country_province.

cumCases = pd.Series()

cumDeath = pd.Series()

for region in train.country_province.unique():

    #print(region)

    cumCases = pd.concat([cumCases,train.ConfirmedCases[train.country_province==region].cumsum()])

    cumDeath = pd.concat([cumDeath,train.Fatalities[train.country_province==region].cumsum()])

    

print(len(cumCases), len(cumDeath), train.shape[0])

train_cum = pd.concat([train,cumCases,cumDeath], axis=1)

print(train.shape, train_cum.shape)

train_cum.rename(columns={0:'cumCases', 1:'cumDeath'}, inplace=True)
train_cum.head(73)
# 3. Remove the zero case days for each country_province.

train_filtered = train_cum[train_cum.ConfirmedCases > 0.0]

train_filtered.head(50)
train_cum.to_csv('train_cum.csv', index=False)

train_filtered.to_csv('train_filter.csv', index=False)
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import StackingRegressor

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.linear_model import LassoCV

from sklearn.linear_model import RidgeCV



# create models

estimators = [('Random Forest',RandomForestRegressor(random_state=42)),

         ('Lasso',LassoCV()),

         ('Gradient Boosting', HistGradientBoostingRegressor(random_state=0))]

stackingRegressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
import time

import numpy as np

from sklearn.model_selection import cross_validate, cross_val_predict
