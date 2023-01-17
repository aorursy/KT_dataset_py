"""author s_agnik1511"""

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.offline as py

import plotly.graph_objs as go



import gc

import sys



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder



# Models Libs

from xgboost import XGBRegressor



import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)
"""author s_agnik1511"""

# Csv Files

train_data_path = '../input/house-prices-advanced-regression-techniques/train.csv'

test_data_path = '../input/house-prices-advanced-regression-techniques/test.csv'
"""author s_agnik1511"""

# Read csv

train_df = pd.read_csv(train_data_path)

test_df = pd.read_csv(test_data_path)
# Look to shape's

print('Train Shape:', train_df.shape)

print('Test Shape:', test_df.shape)
"""author s_agnik1511"""

# See the raw data, first 10

train_df.head()
"""author s_agnik1511"""

year_stations_df = train_df[['SalePrice','MoSold']].copy()



def setStation(month):

    if month in (1,2,3):

        return "Summer"

    if month in (4,5,6):

        return "Autumn"

    if month in (7,8,9):

        return "Winter"

    return "Spring"

    



year_stations_df['yearStation'] = year_stations_df.MoSold.apply(lambda x: setStation(x));



year_stations_df.sort_values(by='SalePrice', inplace=True)



trace = go.Box(

    x = year_stations_df.yearStation,

    y = year_stations_df.SalePrice

)



data = [trace]



layout = go.Layout(title="Prices x Year Station",

                  yaxis={'title':'Sale Price'},

                  xaxis={'title':'Year Station'})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
"""author s_agnik1511"""

year_stations_gp_df = year_stations_df.groupby('yearStation')['SalePrice'].count().reset_index()

year_stations_gp_df = pd.DataFrame({'yearStation': year_stations_gp_df.yearStation,

                                   'CountHouse': year_stations_gp_df.SalePrice})

year_stations_gp_df.sort_values(by='CountHouse', inplace=True)
"""author s_agnik1511"""

trace = go.Bar(

    x = year_stations_gp_df.yearStation,

    y = year_stations_gp_df.CountHouse

)

data = [trace]

layout = go.Layout(title="Count House x Year Station",

                  yaxis={'title':'Count House'},

                  xaxis={'title':'Year Station'})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
"""author s_agnik1511"""

def labelStation(x):

    if x=="Summer":

        return 1

    if x=="Autumn":

        return 2

    if x=="Winter":

        return 3

    return 4

year_stations_df['labelStation']=year_stations_df.yearStation.apply(lambda x:labelStation(x))

df_corr_year_stations = year_stations_df.corr()

df_corr_year_stations
"""author s_agnik1511"""

year_stations_sorted_df = year_stations_df.sort_values(by='MoSold')

year_stations_sorted_gp_df = year_stations_df.groupby('MoSold')['SalePrice'].count().reset_index();
"""author s_agnik1511"""

df = year_stations_sorted_gp_df

trace = go.Scatter(

    x = df.MoSold,

    y = df.SalePrice,

    mode = 'markers+lines',

    line_shape='spline'

)



data = [trace]



layout = go.Layout(title="Prices by month's",

                  yaxis={'title':'Sale Price'},

                  xaxis={'title':'Month sold', 'zeroline':False})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
#train_df



trace = go.Scatter(

    x = train_df.LotArea,

    y = train_df.SalePrice,

    mode = 'markers'

)



data = [trace]



layout = go.Layout(title="Lot Area x Sale Price",

                  yaxis={'title':'Sale Price'},

                  xaxis={'title':'Lot Area'})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
trace = go.Box(

    y = train_df.SalePrice,

    name = 'Sale Price'

)



data = [trace]



layout = go.Layout(title="Distribuiton Sale Price",

                  yaxis={'title':'Sale Price'})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
trace = go.Box(

    y = train_df.LotArea,

    name = 'Lot Area'

)



data = [trace]



layout = go.Layout(title="Distribuiton Lot Area",

                  yaxis={'title':'Lot Area'})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
"""author s_agnik1511"""

lotarea_saleprice_df = train_df[['SalePrice', 'LotArea']]

lotarea_saleprice_df.corr()
"""author s_agnik1511"""

train_df = train_df.drop(train_df.loc[(train_df['LotArea'] > 100000)].index)

train_df = train_df.drop(train_df.loc[(train_df['SalePrice'] > 500000)].index)
"""author s_agnik1511"""

trace = go.Scatter(

    x = train_df.LotArea,

    y = train_df.SalePrice,

    mode = 'markers'

)

data = [trace]

layout = go.Layout(title="Lot Area x Sale Price",

                  yaxis={'title':'Sale Price'},

                  xaxis={'title':'Lot Area'})

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
"""author s_agnik1511"""

def setStation(month):

    if month in (1,2,3):

        return "Summer"

    if month in (4,5,6):

        return "Autumn"

    if month in (7,8,9):

        return "Winter"

    return "Spring"

train_df['yearStation'] = train_df.MoSold.apply(lambda x: setStation(x));

test_df['yearStation'] = test_df.MoSold.apply(lambda x: setStation(x));
"""author s_agnik1511"""

# Set y (Target)

y = np.log(train_df.SalePrice)



X = train_df.copy()
"""author s_agnik1511"""

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

X.drop(['SalePrice','OverallQual'], axis=1, inplace=True)

X_test = test_df.copy();
X.head()
"""author s_agnik1511"""

X_val = X.isnull().sum() * 100 / len(X)

X_val.loc[X_val > 50.00]
"""author s_agnik1511"""

colls= [col for col in X.columns if X[col].isnull().sum() * 100 / len(X) > 50.00]

for col in colls:

    X[col].fillna("None")

    X_test[col].fillna("None")
"""author s_agnik1511"""

print('Train Shape:', X.shape)

print('Test Shape:', X_test.shape)
"""author s_agnik1511"""

categorical_cols = [cname for cname in X.columns

if X[cname].dtype == "object"            

]

numerical_cols = [cname for cname in X.columns

if X[cname].dtype in ['int64', 'float64']]
"""author s_agnik1511"""

my_cols = categorical_cols + numerical_cols

X = X[my_cols].copy()

X_test = X_test[my_cols].copy()

X.head()
"""author s_agnik1511"""

x_cat_unique_values  = [col for col in X[categorical_cols].columns if len(X[col].unique()) <= 5]

dict_diff_onehot = set(categorical_cols) - set(x_cat_unique_values)

one_hot_cols = list(dict_diff_onehot)
"""author s_agnik1511"""

for col in numerical_cols:

    X['{}_{}'.format(col,2)] = X[col]**2

    X_test['{}_{}'.format(col,2)] = X_test[col]**2

    X['{}_{}'.format(col,3)] = X[col]**3

    X_test['{}_{}'.format(col,3)] = X_test[col]**3
X.head()
"""author s_agnik1511"""

labelEncoder = LabelEncoder()

for col in x_cat_unique_values:

    x_unique = X[col].unique();

    x_test_unique = X_test[col].unique();

    union_uniques = list(x_unique) + list(x_test_unique)

    uniques = list(dict.fromkeys(union_uniques));

    labelEncoder.fit(uniques);

    X[col] = labelEncoder.transform(X[col].astype(str))

    X_test[col] = labelEncoder.transform(X_test[col].astype(str))
"""author s_agnik1511"""

simple_imp = SimpleImputer(missing_values=np.nan, strategy='constant')

X['GarageYrBlt'] = simple_imp.fit_transform(X[['GarageYrBlt']])

X_test['GarageYrBlt'] = simple_imp.transform(X_test[['GarageYrBlt']])
"""author s_agnik1511"""

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import learning_curve

from math import sqrt

from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor
"""author s_agnik1511"""

X.head()
"""author s_agnik1511"""

numerical_transformer = Pipeline(steps=[

  ('imputer', SimpleImputer(strategy='median')),

  ('scaler', StandardScaler())

])

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, one_hot_cols)

    ]

)

pipeline = Pipeline(

    steps=[

        ('preprocessor', preprocessor),

    ]

)

X_train_fit = pipeline.fit_transform(X)
"""author s_agnik1511"""

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear','dart']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]

base_score=[0.25,0.5,0.75,1]

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }

random_cv = RandomizedSearchCV(estimator=XGBRegressor(),

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_squared_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)

random_cv.fit(X_train_fit,y)
random_cv.best_estimator_
"""author s_agnik1511"""

modelGb =  XGBRegressor(base_score=0.25, booster='dart', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.1, max_delta_step=0,

             max_depth=2, min_child_weight=3, missing=None, n_estimators=500,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

modelCat = CatBoostRegressor(depth=3,learning_rate=0.014,iterations=7600,verbose=False,task_type = 'GPU')
"""author s_agnik1511"""

kf = KFold(5, shuffle=True, random_state=0)

for linhas_treino, linhas_valid in kf.split(X_train_fit):

    X_train, X_valid = X_train_fit[linhas_treino], X_train_fit[linhas_valid];

    y_train, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid];

    model = modelGb;

    model.fit(X_train, y_train);

    preds = model.predict(X_valid)

    print('RMSE:', np.sqrt(mean_squared_error(y_valid, preds)),'\n');


"""author s_agnik1511"""

X_train, X_valid, y_train, y_valid = train_test_split(X_train_fit, y, random_state = 1, train_size=0.8, test_size=0.2)
"""author s_agnik1511"""

# Define Model

model = modelGb;

model.fit(X_train, y_train);

preds = model.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds),'\n');

print('RMSE:', np.sqrt(mean_squared_error(y_valid, preds)),'\n');
"""author s_agnik1511"""

X_test_fit = pipeline.transform(X_test)
"""author s_agnik1511"""

preds_test = model.predict(X_test_fit)

output = pd.DataFrame({'Id': X_test.Id, 'SalePrice': np.exp(preds_test)})

output.to_csv('sample_sub_#3.csv', index=False)

output.head()