# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
PATH_WEEK1='/kaggle/input/covid19-global-forecasting-week-1'
df_train = pd.read_csv(f'{PATH_WEEK1}/train.csv')
df_test = pd.read_csv(f'{PATH_WEEK1}/test.csv')
df_train.head()
df_train.tail()
df_train.rename(columns={'Country/Region':'Country'}, inplace=True)

df_test.rename(columns={'Country/Region':'Country'}, inplace=True)



df_train.rename(columns={'Province/State':'State'}, inplace=True)

df_test.rename(columns={'Province/State':'State'}, inplace=True)
df_test.head()
df_train.info()
df_test.info()
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
df_train.info()
df_train.describe()
df_test.describe()
import plotly.express as px

pxdf = px.data.gapminder()



country_isoAlpha = pxdf[['country', 'iso_alpha']].drop_duplicates()

country_isoAlpha.rename(columns = {'country':'Country'}, inplace=True)

country_isoAlpha.set_index('Country', inplace=True)

country_map = country_isoAlpha.to_dict('index')
def getCountryIsoAlpha(country):

    try:

        return country_map[country]['iso_alpha']

    except:

        return country
df_train['iso_alpha'] = df_train['Country'].apply(getCountryIsoAlpha)

df_train.info()
df_train.isna().sum()
df_train.Country.unique()
df_plot = df_train.loc[:,['Date', 'Country', 'ConfirmedCases']]

df_plot.loc[:, 'Date'] = df_plot.Date.dt.strftime("%Y-%m-%d")

df_plot.loc[:, 'Size'] = np.where(df_plot['Country'].isin(['China', 'Italy']), df_plot['ConfirmedCases'], df_plot['ConfirmedCases']*100)

fig = px.scatter_geo(df_plot.groupby(['Date', 'Country']).max().reset_index(),

                     locations="Country",

                     locationmode = "country names",

                     hover_name="Country",

                     color="ConfirmedCases",

                     animation_frame="Date", 

                     size='Size',

                     #projection="natural earth",

                     title="Rise of Coronavirus Confirmed Cases")

fig.show()
df_train.drop(columns='iso_alpha', inplace=True)

df_train
X_Train = df_train.copy()



X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%Y%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)



X_Train.drop(columns=['ConfirmedCases', 'Fatalities'], inplace=True)

X_Train.drop(columns=['Id', 'State', 'Country', 'Lat'], inplace=True)

X_Train.head()
X_Train.head()
X_Train.head()
y1_Train = df_train.ConfirmedCases
from warnings import filterwarnings

filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, SGDRegressor

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.svm import LinearSVR, NuSVR, SVR

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor 



MODELS = {"Linear_Reg": LinearRegression(), "KNN_Reg": KNeighborsRegressor(), "LinearSVR_Reg": LinearSVR(), "DecisionTree_Reg": DecisionTreeRegressor(), "DecisionTree_Class": DecisionTreeClassifier(), "ExtraTree_Reg": ExtraTreeRegressor(), "RandomForest_Reg": RandomForestRegressor(), "GB_Reg": GradientBoostingRegressor(), "XGB_Reg": XGBRegressor() }

# "SDG_Reg": SGDRegressor(), "RN_Reg": RadiusNeighborsRegressor(), "NuSVR_Reg": NuSVR(), "SVR_Reg": SVR(),
from sklearn.model_selection import GridSearchCV

import time



model = XGBRegressor()
#param_grid = {"criterion": ["mse", "mae"], "min_samples_split": [10, 20], "max_depth": [2, 6], "min_samples_leaf": [20, 40], "max_leaf_nodes": [5, 20]} #DTR

#param_grid = {"criterion": ["mae"], 'max_depth': [2], 'max_leaf_nodes': [5], 'min_samples_leaf': [20], 'min_samples_split': [10]} #DTR best

#param_grid = {'criterion':['gini'], 'max_depth': np.arange(2,5), 'min_samples_leaf': range(1,5)} #DTC

#param_grid = {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1} #DTC best

#param_grid = {'n_neighbors':range(1,2,4), 'leaf_size':[4,5,6], 'weights':['uniform', 'distance'], 'algorithm':['auto', 'ball_tree','kd_tree','brute']} #KNR

#param_grid = {'algorithm': ['auto'], 'leaf_size': [4], 'n_neighbors': [1], 'weights': ['distance']} #KNR best

#param_grid = {'nthread':[4], 'objective':['reg:linear'], 'learning_rate': [.03, 0.05], 'max_depth': [5, 6], 'min_child_weight': [4], 'silent': [1], 'subsample': [0.7], 'colsample_bytree': [0.7], 'n_estimators': [500, 1000]} #XGBR

param_grid = {'n_estimators': [1250]}
start = time.time()

grid_cv = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error')

grid_cv.fit(X_Train, y1_Train)

print (f'{type(model).__name__} Hyper Paramter Tuning took a Time: {time.time() - start}')
print("Mean Squared Error: {}".format(grid_cv.best_score_))

print("Best Hyperparameters:\n{}".format(grid_cv.best_params_))



#df_dtr = pd.DataFrame(data=grid_cv.cv_results_)

#print(df_dtr.head())
X_Test = df_test.copy()



X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%Y%m%d")

X_Test["Date"]  = X_Test["Date"].astype(int)



#X_Test.drop(columns=['ConfirmedCases', 'Fatalities'], inplace=True)

X_Test.drop(columns=['ForecastId', 'State', 'Country', 'Lat'], inplace=True)

X_Test.head()

y1_best_dtr_model = grid_cv.best_estimator_

y1_pred = y1_best_dtr_model.predict(X_Test)
y1_pred = y1_pred.round(0)
y1_pred
from sklearn.metrics import r2_score

print(r2_score(y1_Train, y1_best_dtr_model.predict(X_Train))) 
y2_Train = df_train.Fatalities
start = time.time()

grid_cv = GridSearchCV(model, param_grid, cv=10)

grid_cv.fit(X_Train, y2_Train)

print (f'Decision Tree Regressor Hyper Paramter Tuning took a Time: {time.time() - start}')
print("R-Squared::{}".format(grid_cv.best_score_))

print("Best Hyperparameters::\n{}".format(grid_cv.best_params_))



#df_dtr = pd.DataFrame(data=grid_cv.cv_results_)

#print(df_dtr.head())
y2_best_dtr_model = grid_cv.best_estimator_

y2_pred = y2_best_dtr_model.predict(X_Test)
y2_pred = y2_pred.round(0)
y2_pred
df_sub = pd.read_csv(f'{PATH_WEEK1}/submission.csv')
df_sub.head()
df_sub.info()
df = pd.DataFrame({"ForecastId": df_test.ForecastId, "ConfirmedCases": y1_pred, "Fatalities": y2_pred})

df.head()
df.to_csv('submission.csv', index=False)