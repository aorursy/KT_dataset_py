# My default libs

import numpy as np

import pandas as pd



# Data viz libs

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.offline as py

import plotly.graph_objs as go



# Sklearn libs

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder



# Models Libs

from xgboost import XGBRegressor



# Disable warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# Files path

train_data_path = '../input/house-prices-advanced-regression-techniques/train.csv'

test_data_path = '../input/house-prices-advanced-regression-techniques/test.csv'
# Getting the dataset's

train_df = pd.read_csv(train_data_path)

test_df = pd.read_csv(test_data_path)
# Look to shape's

print('Train dataset have ', train_df.shape[0], ' lines and ', train_df.shape[1], ' columns')

print('Test dataset have ', test_df.shape[0], ' lines and ', test_df.shape[1], ' columns')
# See the raw data, first 10 lines

train_df.head()
year_seasons_df = train_df[['SalePrice','MoSold']].copy()



def setSeason(month):

    if month in (6,7,8):

        return "Summer"

    if month in (11,10,9):

        return "Autumn"

    if month in (12,1,2):

        return "Winter"

    return "Spring"

    



year_seasons_df['yearSeason'] = year_seasons_df.MoSold.apply(lambda x: setSeason(x));



year_seasons_df.sort_values(by='SalePrice', inplace=True)



trace = go.Box(

    x = year_seasons_df.yearSeason,

    y = year_seasons_df.SalePrice

)



data = [trace]



layout = go.Layout(title="Prices x Year Season",

                  yaxis={'title':'Sale Price'},

                  xaxis={'title':'Year Season'})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
year_seasons_gp_df = year_seasons_df.groupby('yearSeason')['SalePrice'].count().reset_index()



year_seasons_gp_df = pd.DataFrame({'yearSeason': year_seasons_gp_df.yearSeason,

                                   'CountHouse': year_seasons_gp_df.SalePrice})



year_seasons_gp_df.sort_values(by='CountHouse', inplace=True)
trace = go.Bar(

    x = year_seasons_gp_df.yearSeason,

    y = year_seasons_gp_df.CountHouse

)



data = [trace]



layout = go.Layout(title="Count House x Year Station",

                  yaxis={'title':'Count House'},

                  xaxis={'title':'Year Station'})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
def labelSeason(x):

    if x == "Summer":

        return 1

    if x == "Autumn":

        return 2

    if x == "Winter":

        return 3

    return 4





year_seasons_df['labelSeason'] = year_seasons_df.yearSeason.apply(lambda x: labelSeason(x))



df_corr_year_seasons = year_seasons_df.corr()



df_corr_year_seasons
year_seasons_sorted_df = year_seasons_df.sort_values(by='MoSold')



year_seasons_sorted_gp_df = year_seasons_df.groupby('MoSold')['SalePrice'].count().reset_index();
df = year_seasons_sorted_gp_df



trace = go.Scatter(

    x = df.MoSold,

    y = df.SalePrice,

    mode = 'markers+lines',

    line_shape='spline'

)



data = [trace]



layout = go.Layout(title="Sales by month's",

                  yaxis={'title':'Count House'},

                  xaxis={'title':'Month sold', 'zeroline':False})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
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
lotarea_saleprice_df = train_df[['SalePrice', 'LotArea']]



lotarea_saleprice_df.corr()
train_df = train_df.drop(train_df.loc[(train_df['LotArea'] > 70000)].index)

train_df = train_df.drop(train_df.loc[(train_df['SalePrice'] > 500000)].index)
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
# Set y (Target)

y = np.log(train_df.SalePrice)



X = train_df.copy()



X_test = test_df.copy();
X['AreaUtil'] = X['LotArea'] - (X['MasVnrArea'] + X['GarageArea'] + X['PoolArea'])

X_test['AreaUtil'] = X_test['LotArea'] - (X_test['MasVnrArea'] + X_test['GarageArea'] + X_test['PoolArea'])
X['HavePool'] = X['PoolArea'] > 0

X_test['HavePool'] = X_test['PoolArea'] > 0
X['GarageCars2'] = X['GarageCars']**2

X['GarageCarsSQRT'] = np.sqrt(X['GarageCars'])

X['GarageArea'] = X['GarageArea']**2

X['GarageAreaSQRT'] = np.sqrt(X['GarageArea'])

X['LotArea2'] = X['LotArea']**2

X['LotAreaSQRT'] = np.sqrt(X['LotArea'])

X['AreaUtil2'] = X['AreaUtil']**2

X['AreaUtilSQRT'] = np.sqrt(X['AreaUtil'])

X['GrLivArea2'] = X['GrLivArea']**2

X['GrLivAreaSQRT'] = np.sqrt(X['GrLivArea'])



X_test['GarageCars2'] = X_test['GarageCars']**2

X_test['GarageCarsSQRT'] = np.sqrt(X_test['GarageCars'])

X_test['GarageArea'] = X_test['GarageArea']**2

X_test['GarageAreaSQRT'] = np.sqrt(X_test['GarageArea'])

X_test['LotArea2'] = X_test['LotArea']**2

X_test['LotAreaSQRT'] = np.sqrt(X_test['LotArea'])

X_test['AreaUtil2'] = X_test['AreaUtil']**2

X_test['AreaUtilSQRT'] = np.sqrt(X_test['AreaUtil'])

X_test['GrLivArea2'] = X_test['GrLivArea']**2

X_test['GrLivAreaSQRT'] = np.sqrt(X_test['GrLivArea'])
corrmat = X.corr()



top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0]



# most correlated features

if 1 == 1:

    plt.figure(figsize=(30,15))

    g = sns.heatmap(X[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# Remove row with missing target

X.dropna(axis=0, subset=['SalePrice'], inplace=True)



# Drop target.

X.drop(['SalePrice'], axis=1, inplace=True)



X.drop(['OverallQual'], axis=1, inplace=True)

X.head()
#cols_sem_muitos_dados = [col for col in X.columns if X[col].isnull().sum() * 100 / len(X) > 50.00]



cols_sem_muitos_dados = [col for col in X.columns if X[col].isnull().any()]



for col in cols_sem_muitos_dados:

    X[col].fillna("None")

    X_test[col].fillna("None")



#X.drop(cols_sem_muitos_dados, axis=1, inplace=True)

#X_test.drop(cols_sem_muitos_dados, axis=1, inplace=True)
X.head()
print('Train Shape:', X.shape)

print('Test Shape:', X_test.shape)
# Get categorical cols WITH type OBJECT (like string)

categorical_cols = [cname for cname in X.columns

if X[cname].dtype == "object"            

]



# Get numerical cols

numerical_cols = [cname for cname in X.columns

if X[cname].dtype in ['int64', 'float64']]
# Merge all cols

my_cols = categorical_cols + numerical_cols

X = X[my_cols].copy()

X_test = X_test[my_cols].copy()



# Let's see the first 5 cols

X.head()
one_hot_cols = categorical_cols



if 1 == 0:

    

    x_cat_unique_values  = [col for col in X[categorical_cols].columns if len(X[col].unique()) <= 10]



    dict_diff_onehot = set(categorical_cols) - set(x_cat_unique_values)



    one_hot_cols = x_cat_unique_values



    labelEncoder = LabelEncoder()



    for col in list(dict_diff_onehot):

        x_unique = X[col].unique();

        x_test_unique = X_test[col].unique();



        union_uniques = list(x_unique) + list(x_test_unique)



        uniques = list(dict.fromkeys(union_uniques));



        labelEncoder.fit(uniques);



        X[col] = labelEncoder.transform(X[col].astype(str))

        X_test[col] = labelEncoder.transform(X_test[col].astype(str))

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import learning_curve

from math import sqrt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold

from sklearn.decomposition import SparsePCA

from sklearn.preprocessing import MaxAbsScaler

from catboost import CatBoostRegressor
X.head()
# Numerical data

numerical_transformer = Pipeline(steps=[

  ('imputer', SimpleImputer(strategy='mean')),

  ('scaler', MaxAbsScaler())

])



# Categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, one_hot_cols)

    ]

)



# Bundle preprocessing and modeling the pipeline

pipeline = Pipeline(

    steps=[

        ('preprocessor', preprocessor),

    ]

)



# Preprocessing of training data

X_train_fit = pipeline.fit_transform(X)

X_test_fit = pipeline.transform(X_test)
#print("How much features we have now =  "+str(len(X_train_fit[0])))
if 1 == 1:

    from sklearn.feature_selection import RFE

    

    nfeature = 200

    

    trans = RFE(XGBRegressor(random_state=0,objective="reg:squarederror"), n_features_to_select=nfeature)

    

    X_train_fit = trans.fit_transform(X_train_fit, y)

    X_test_fit = trans.transform(X_test_fit)
if 1 == 0:

    from sklearn.model_selection import RandomizedSearchCV

   

    try:

        nfeature

    except NameError:

        nfeature = len(X_train_fit[0])

        

    n_estimators = range(100,2000,200)

    max_depth = [2, 3, 5, 10, 15]

    booster=['gbtree','gblinear','dart']

    learning_rate=[0.05,0.1,0.012,0.013,0.015,0.016,0.020,0.025,0.2,0.3,0.5]

    min_child_weight=[1,2,3,4]

    base_score=[0.25,0.5,0.75,1]

    max_features=range(1, nfeature)



    # Define the grid of hyperparameters to search

    hyperparameter_grid = {

        'n_estimators': n_estimators,

        'max_depth':max_depth,

        'learning_rate':learning_rate,

        'min_child_weight':min_child_weight,

        'booster':booster,

        'base_score':base_score,

        'max_features':max_features

        }



    random_cv = RandomizedSearchCV(estimator=XGBRegressor(random_state=0,objective="reg:squarederror"),

                param_distributions=hyperparameter_grid,

                cv=5, n_iter=25,

                scoring = 'neg_mean_squared_error',n_jobs = 4,

                verbose = 5,

                return_train_score = False,

                random_state=0)



    random_cv.fit(X_train_fit,y)

    

    model = random_cv.best_estimator_

    

    print(random_cv.best_estimator_)
if 1 == 1:

    model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.015, max_delta_step=0,

             max_depth=3, max_features=97, min_child_weight=1, missing=None,

             n_estimators=1100, n_jobs=1, nthread=None,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,

             subsample=1, verbosity=1)

    
#len(X_train_fit[0])
kf = KFold(5, shuffle=True, random_state=0)



for linhas_treino, linhas_valid in kf.split(X_train_fit):

    X_train, X_valid = X_train_fit[linhas_treino], X_train_fit[linhas_valid];

    y_train, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid];

    

    # Define Model

    model.fit(X_train, y_train);



    # Preprocessing of validation data, get predictions

    

    preds = model.predict(X_valid)

    

    print('MAE:', mean_absolute_error(np.exp(y_valid), np.exp(preds)),'\n');

    print('RMSE:', np.sqrt(mean_squared_error(y_valid, preds)),'\n');
if 1 == 1:

    # Split data with validation data and train data

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_fit, y, random_state = 0, train_size=0.8)

    

    model.fit(X_train, y_train);



    # Preprocessing of validation data, get predictions

    preds = model.predict(X_valid)



    print('MAE:', mean_absolute_error(np.exp(y_valid), np.exp(preds)),'\n');

    print('RMSE:', np.sqrt(mean_squared_error(y_valid, preds)),'\n');
preds_test = model.predict(X_test_fit)



# Create OutPut Data

output = pd.DataFrame({'Id': X_test.Id, 'SalePrice': np.exp(preds_test)})



# To CSV

output.to_csv('submission.csv', index=False)



# Show me the data!

output.head()