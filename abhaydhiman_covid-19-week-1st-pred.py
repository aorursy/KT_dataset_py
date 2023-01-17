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

import matplotlib.pyplot as plt

import seaborn as sns

from math import cos, sin
train_df = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test_df = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')



df = [train_df, test_df]
train_df.shape
train_df.head()
train_df.isnull().sum()
arr1 = train_df['Country/Region'].unique()
arr2 = [i for i in range(163)]
country_dict = dict(zip(arr1, arr2))
country_dict
for dataset in df:

    dataset['Country_int'] = dataset['Country/Region'].map(country_dict).astype(int)
train_df.head()
type(train_df.Date)
for dataset in df:

    dataset['Date'] = pd.to_datetime(dataset['Date'])
for dataset in df:

    dataset['Day'] = dataset.Date.apply(lambda x: x.day)

    dataset['Month'] = dataset.Date.apply(lambda x: x.month)
train_df.head()
for j in train_df['Country/Region'].unique():

    l = train_df[(train_df['Country/Region'] == j)]

    len_state = len(l['Province/State'].unique())

    

    if len_state != 1:

        arr_state1 = l['Province/State'].unique()

        arr_state2 = [i for i in range(1,len_state+1)]

        state_dict = dict(zip(arr_state1, arr_state2))



        for dataset in df:

            dataset['Province/State'] = dataset['Province/State'].map(state_dict)
train_df[train_df['Country/Region'] == 'Australia'].head()
for dataset in df:

    dataset['Province/State'].fillna(0, inplace=True)
train_df.head()
for dataset in df:

    dataset['X_Cord'] = (dataset['Lat'].apply(cos)) * (dataset['Long'].apply(cos))

    dataset['Y_Cord'] = (dataset['Lat'].apply(cos)) * (dataset['Long'].apply(sin))    
train_df[train_df['Country/Region'] == 'Afghanistan'].head()
test_df.head()
train_df.info()
for dataset in df:

    dataset['Province/State'] = dataset['Province/State'].astype(int)
train_df['ConfirmedCases'] = train_df['ConfirmedCases'].astype(int)

train_df['Fatalities'] = train_df['Fatalities'].astype(int)
train_df.info()
import seaborn as sns
corr = train_df.corr()

plt.figure(figsize=(11,7))

sns.heatmap(corr, annot=True)
X = train_df.drop(['Id', 'Country/Region', 'Lat', 'Long', 'Date', 'ConfirmedCases', 'Fatalities', 'Province/State'], axis=1)

y1 = train_df['ConfirmedCases']

y2 = train_df['Fatalities']

X_test = test_df.drop(['ForecastId', 'Country/Region', 'Lat', 'Long', 'Date', 'Province/State'], axis=1)
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
lin_reg_regressor = LinearRegression()

lin_reg_regressor.fit(X, y1)

lin_reg_regressor.score(X, y1)
KNN_regressor = KNeighborsRegressor()

KNN_regressor.fit(X, y1)

KNN_regressor.score(X, y1)
tree_regressor = DecisionTreeRegressor()

tree_regressor.fit(X, y1)

tree_regressor.score(X, y1)
Random_Forest_regressor = RandomForestRegressor()

Random_Forest_regressor.fit(X, y1)

Random_Forest_regressor.score(X, y1)
XGB_regressor = XGBRegressor()

XGB_regressor.fit(X, y1)

XGB_regressor.score(X, y1)
param_test = {

    'max_depth': range(3, 12, 2),

    'min_child_weight': range(1, 6, 2),

    'gamma': [i/10.0 for i in range(0,5)],

#     'subsample': [i/10.0 for i in range(6,10)],

#     'colsample_bytree': [i/10.0 for i in range(6,10)],

#     'reg_alpha': [0, 0.001, 0.005, 0.01, 1, 100],

    'learning_rate': [0.1, 0.2, 0.3],

    'n_estimators': [100, 400, 600, 900, 1100]

}
gsearch = GridSearchCV(estimator = XGBRegressor(), 

                       param_grid = param_test,

                       scoring='neg_root_mean_squared_error',

                       n_jobs=-1,

                       cv=5)
# gsearch.fit(X, y1)
gsearch.best_estimator_
XGB_regressor=XGBRegressor(base_score=0.5, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0.0, gpu_id=-1,

             importance_type='gain', learning_rate=0.1, max_delta_step=0, max_depth=3,

             min_child_weight=1, n_estimators=100, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1)





XGB_regressor.fit(X, y1)

XGB_regressor.score(X, y1)
ConfirmedCasesPred = XGB_regressor.predict(X_test)

ConfirmedCasesPred = pd.DataFrame(ConfirmedCasesPred, columns=['ConfirmedCases'])
XGB_regressor.fit(X, y2)

XGB_regressor.score(X, y2)
FatalitiesPred = XGB_regressor.predict(X_test)

FatalitiesPred = pd.DataFrame(FatalitiesPred, columns=['Fatalities'])
ForecastId = test_df.ForecastId

ForecastId = pd.DataFrame(ForecastId)
pred_file = pd.concat([ForecastId, ConfirmedCasesPred, FatalitiesPred], axis=1)
pred_file.head()
pred_file.to_csv('submission.csv', index=False)