import numpy as np

import pandas as pd

pd.set_option('max_columns', 105)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



from scipy import stats

from scipy.stats import skew

from math import sqrt



# plotly

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



# sklearn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, Lasso, ElasticNet, BayesianRidge

from sklearn.kernel_ridge import KernelRidge



from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor



import xgboost as xgb

from xgboost import XGBRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor



from mlxtend.regressor import StackingRegressor



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)

#warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

def get_best_score(grid):

    

    best_score = np.sqrt(-grid.best_score_)

    print(best_score)    

    print(grid.best_params_)

    print(grid.best_estimator_)

    

    return best_score
def plotly_scatter_x_y(df_plot, val_x, val_y):

    

    value_x = df_plot[val_x] 

    value_y = df_plot[val_y]

    

    trace_1 = go.Scatter( x = value_x, y = value_y, name = val_x, 

                         mode="markers", opacity=0.8 )



    data = [trace_1]

    

    plot_title = val_y + " vs. " + val_x

    

    layout = dict(title = plot_title, 

                  xaxis=dict(title = val_x, ticklen=5, zeroline= False),

                  yaxis=dict(title = val_y, side='left'),                                  

                  legend=dict(orientation="h", x=0.4, y=1.0),

                  autosize=False, width=750, height=500,

                 )



    fig = dict(data = data, layout = layout)

    iplot(fig)

def plotly_scatter_x_y_color(df_plot, val_x, val_y, val_z):

    

    value_x = df_plot[val_x] 

    value_y = df_plot[val_y]

    value_z = df_plot[val_z]

    

    trace_1 = go.Scatter( 

                         x = value_x, y = value_y, name = val_x, 

                         mode="markers", opacity=0.8, text=value_z,

                         marker=dict(size=6, color = value_z, 

                                     colorscale='Jet', showscale=True),                        

                        )

                            

    data = [trace_1]

    

    plot_title = val_y + " vs. " + val_x

    

    layout = dict(title = plot_title, 

                  xaxis=dict(title = val_x, ticklen=5, zeroline= False),

                  yaxis=dict(title = val_y, side='left'),                                  

                  legend=dict(orientation="h", x=0.4, y=1.0),

                  autosize=False, width=750, height=500,

                 )



    fig = dict(data = data, layout = layout)

    iplot(fig)
def plotly_scatter_x_y_catg_color(df, val_x, val_y, val_z):

    

    catg_for_colors = sorted(df[val_z].unique().tolist())



    fig = { 'data': [{ 'x': df[df[val_z]==catg][val_x],

                       'y': df[df[val_z]==catg][val_y],    

                       'name': catg, 

                       'text': df[val_z][df[val_z]==catg], 

                       'mode': 'markers',

                       'marker': {'size': 6},

                      

                     } for catg in catg_for_colors       ],

                       

            'layout': { 'xaxis': {'title': val_x},

                        'yaxis': {'title': val_y},                    

                        'colorway' : ['#a9a9a9', '#e6beff', '#911eb4', '#4363d8', '#42d4f4',

                                      '#3cb44b', '#bfef45', '#ffe119', '#f58231', '#e6194B'],

                        'autosize' : False, 

                        'width' : 750, 

                        'height' : 600,

                      }

           }

  

    iplot(fig)
def plotly_scatter3d(data, feat1, feat2, target) :



    df = data

    x = df[feat1]

    y = df[feat2]

    z = df[target]



    trace1 = go.Scatter3d( x = x, y = y, z = z,

                           mode='markers',

                           marker=dict( size=5, color=y,               

                                        colorscale='Viridis',  

                                        opacity=0.8 )

                          )

    data = [trace1]

    camera = dict( up=dict(x=0, y=0, z=1),

                   center=dict(x=0, y=0, z=0.0),

                   eye=dict(x=2.5, y=0.1, z=0.8) )



    layout = go.Layout( title= target + " as function of " +  

                               feat1 + " and " + feat2 ,

                        autosize=False, width=700, height=600,               

                        margin=dict( l=15, r=25, b=15, t=30 ) ,

                        scene=dict(camera=camera,

                                   xaxis = dict(title=feat1),

                                   yaxis = dict(title=feat2),

                                   zaxis = dict(title=target),                                   

                                  ),

                       )



    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
print("df_train.shape : " , df_train.shape)

print("*"*50)

print("df_test.shape  : " , df_test.shape)
df_train.info()
df_train.head()
# dropping the column "Id" since it is not useful for predicting SalePrice

df_train.drop('Id',axis=1,inplace=True )

id_test = df_test['Id']                      # for submissions

df_test.drop('Id',axis=1,inplace=True )
df_train.describe().transpose()
df_train.describe(include = ['O']).transpose()
df_train_null = pd.DataFrame()

df_train_null['missing'] = df_train.isnull().sum()[df_train.isnull().sum() > 0].sort_values(ascending=False)



df_test_null = pd.DataFrame(df_test.isnull().sum(), columns = ['missing'])

df_test_null = df_test_null.loc[df_test_null['missing'] > 0]
trace1 = go.Bar(x = df_train_null.index, 

                y = df_train_null['missing'],

                name="df_train", 

                text = df_train_null.index)



trace2 = go.Bar(x = df_test_null.index, 

                y = df_test_null['missing'],

                name="df_test", 

                text = df_test_null.index)



data = [trace1, trace2]



layout = dict(title = "NaN in test and train", 

              xaxis=dict(ticklen=10, zeroline= False),

              yaxis=dict(title = "number of rows", side='left', ticklen=10,),                                  

              legend=dict(orientation="v", x=1.05, y=1.0),

              autosize=False, width=750, height=500,

              barmode='stack'

              )



fig = dict(data = data, layout = layout)

iplot(fig)
df_train.drop(['PoolQC', 'FireplaceQu', 'Fence', 

               'Alley', 'MiscFeature'], axis=1, inplace=True)

df_test.drop(['PoolQC', 'FireplaceQu', 'Fence',

               'Alley', 'MiscFeature'], axis=1, inplace=True)
numerical_columns = df_train.select_dtypes(exclude=['object']).columns.tolist()

print(numerical_columns)
df_train["SalePrice_Log"] = np.log1p(df_train["SalePrice"])
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, 

                          subplot_titles=["SalePrice", "SalePriceLog"])





trace_1 = go.Histogram(x=df_train["SalePrice"], name="SalePrice")

trace_2 = go.Histogram(x=df_train["SalePrice_Log"], name="SalePriceLog")



fig.append_trace(trace_1, 1, 1)

fig.append_trace(trace_2, 1, 2)



iplot(fig)
from scipy.stats import skew, kurtosis

print(df_train["SalePrice"].skew(),"   ", df_train["SalePrice"].kurtosis())

print(df_train["SalePrice_Log"].skew(),"  ", df_train["SalePrice_Log"].kurtosis())
df_corr = df_train.corrwith(df_train['SalePrice']).abs().sort_values(ascending=False)[2:]



data = go.Bar(x=df_corr.index, 

              y=df_corr.values )

       

layout = go.Layout(title = 'Correlation to Sale Price', 

                   xaxis = dict(title = ''), 

                   yaxis = dict(title = 'correlation'),

                   autosize=False, width=750, height=500,)



fig = dict(data = [data], layout = layout)

iplot(fig)
plotly_scatter_x_y(df_train, 'GrLivArea', 'SalePrice')
# outliers GrLivArea

outliers_GrLivArea = df_train.loc[(df_train['GrLivArea']>4000.0) & (df_train['SalePrice']<300000.0)]

outliers_GrLivArea[['GrLivArea' , 'SalePrice']]
df_train['sum_1SF_2SF_LowQualSF'] =  df_train['1stFlrSF'] + df_train['2ndFlrSF'] + df_train['LowQualFinSF']  

df_test['sum_1SF_2SF_LowQualSF'] =  df_test['1stFlrSF'] + df_test['2ndFlrSF'] + df_test['LowQualFinSF'] 

print(sum(df_train['sum_1SF_2SF_LowQualSF'] != df_train['GrLivArea']))

print(sum(df_test['sum_1SF_2SF_LowQualSF'] != df_test['GrLivArea']))
df_train.drop('sum_1SF_2SF_LowQualSF',axis=1,inplace=True )

df_test.drop('sum_1SF_2SF_LowQualSF',axis=1,inplace=True )
df_train['GrLivArea'].corr(df_train['SalePrice'])
(df_train['GrLivArea']-df_train['LowQualFinSF']).corr(df_train['SalePrice'])
y_col_vals = 'SalePrice'

area_features = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

                 'MasVnrArea', 'GarageArea', 'LotArea',

                 'WoodDeckSF', 'OpenPorchSF', 'BsmtFinSF1']

                # 'ScreenPorch'

x_col_vals = area_features
 
nr_rows=3

nr_cols=3



fig = tools.make_subplots(rows=nr_rows, cols=nr_cols, print_grid=False,

                          subplot_titles=area_features )

                                                                

for row in range(1,nr_rows+1):

    for col in range(1,nr_cols+1): 

        

        i = (row-1) * nr_cols + col-1

                   

        trace = go.Scatter(x = df_train[x_col_vals[i]], 

                           y = df_train[y_col_vals], 

                           name=x_col_vals[i], 

                           mode="markers", 

                           opacity=0.8)



        fig.append_trace(trace, row, col,)

 

                                                                                                  

fig['layout'].update(height=700, width=900, showlegend=False,

                     title='SalePrice' + ' vs. Area features')

iplot(fig)                                                
df_train['all_Liv_SF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF'] 

df_test['all_Liv_SF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF'] 



print(df_train['all_Liv_SF'].corr(df_train['SalePrice']))

print(df_train['all_Liv_SF'].corr(df_train['SalePrice_Log']))
df_train['all_SF'] = ( df_train['all_Liv_SF'] + df_train['GarageArea'] + df_train['MasVnrArea'] 

                       + df_train['WoodDeckSF'] + df_train['OpenPorchSF'] + df_train['ScreenPorch'] )

df_test['all_SF'] = ( df_test['all_Liv_SF'] + df_test['GarageArea'] + df_test['MasVnrArea']

                      + df_test['WoodDeckSF'] + df_test['OpenPorchSF'] + df_train['ScreenPorch'] )



print(df_train['all_SF'].corr(df_train['SalePrice']))

print(df_train['all_SF'].corr(df_train['SalePrice_Log']))
df_train['all_SF'].corr(df_train['all_Liv_SF'])
plotly_scatter_x_y(df_train, 'all_SF', 'SalePrice')
outliers_allSF = df_train.loc[(df_train['all_SF']>8000.0) & (df_train['SalePrice']<200000.0)]

outliers_allSF[['all_SF' , 'SalePrice']]
df_train = df_train.drop(outliers_allSF.index)
df_train.corr().abs()[['SalePrice','SalePrice_Log']].sort_values(by='SalePrice', ascending=False)[2:16]
trace = []

for name, group in df_train[["SalePrice", "OverallQual"]].groupby("OverallQual"):

    trace.append( go.Box( y=group["SalePrice"].values, name=name ) )

    

layout = go.Layout(title="OverallQual", 

                   xaxis=dict(title='OverallQual',ticklen=5, zeroline= False),

                   yaxis=dict(title='SalePrice', side='left'),

                   autosize=False, width=750, height=500)



fig = go.Figure(data=trace, layout=layout)

iplot(fig)
outliers_OverallQual_4 = df_train.loc[(df_train['OverallQual']==4) & (df_train['SalePrice']>200000.0)]

outliers_OverallQual_8 = df_train.loc[(df_train['OverallQual']==8) & (df_train['SalePrice']>500000.0)]

outliers_OverallQual_9 = df_train.loc[(df_train['OverallQual']==9) & (df_train['SalePrice']>500000.0)]

outliers_OverallQual_10 = df_train.loc[(df_train['OverallQual']==10) & (df_train['SalePrice']>700000.0)]



outliers_OverallQual = pd.concat([outliers_OverallQual_4, outliers_OverallQual_8, 

                                  outliers_OverallQual_9, outliers_OverallQual_10])
outliers_OverallQual[['OverallQual' , 'SalePrice']]
df_train = df_train.drop(outliers_OverallQual.index)
df_train.corr().abs()[['SalePrice','SalePrice_Log']].sort_values(by='SalePrice', ascending=False)[2:16]
plotly_scatter_x_y_catg_color(df_train, 'all_SF', 'SalePrice', 'OverallQual')
plotly_scatter3d(df_train, 'all_SF', 'OverallQual', 'SalePrice')
print(df_train['OverallQual'].corr(df_train['all_SF']))
print(df_train['OverallCond'].corr(df_train['SalePrice']))

print(df_train['OverallCond'].corr(df_train['SalePrice_Log']))
print(df_train['MSSubClass'].corr(df_train['SalePrice']))

print(df_train['MSSubClass'].corr(df_train['SalePrice_Log']))
 

categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()
def plotly_boxplots_sorted_by_yvals(df, catg_feature, sort_by_target):

    

    df_by_catg   = df.groupby([catg_feature])

    sortedlist_catg_str = df_by_catg[sort_by_target].median().sort_values().keys().tolist()

    

    

    data = []

    for i in sortedlist_catg_str :

        data.append(go.Box(y = df[df[catg_feature]==i][sort_by_target], name = i))



    layout = go.Layout(title = sort_by_target + " vs " + catg_feature, 

                       xaxis = dict(title = catg_feature), 

                       yaxis = dict(title = sort_by_target))



    fig = dict(data = data, layout = layout)

    return fig
fig = plotly_boxplots_sorted_by_yvals(df_train, 'Neighborhood', 'SalePrice')

iplot(fig)
fig = plotly_boxplots_sorted_by_yvals(df_train, 'MSZoning', 'SalePrice')

iplot(fig)
outliers_all = []

df_train = df_train.drop(outliers_all)
# store target as y and y_log:

y , y_log = df_train["SalePrice"] , df_train["SalePrice_Log"]

# drop target from df_train:

df_train.drop(["SalePrice", "SalePrice_Log"] , axis=1, inplace=True)
X_1 = df_train

y_1 = y_log
X_2 = df_train

y_2 = y
from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
numerical_features   = df_train.select_dtypes(exclude=['object']).columns.tolist()

categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()
numerical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer,   numerical_features),

        ('cat', categorical_transformer, categorical_features)])
# LinearRegression

pipe_Linear = Pipeline(

    steps   = [('preprocessor', preprocessor),

               ('Linear', LinearRegression()) ])    

# Ridge

pipe_Ridge = Pipeline(

    steps  = [('preprocessor', preprocessor),

              ('Ridge', Ridge(random_state=5)) ])  

# Huber

pipe_Huber = Pipeline(

    steps  = [('preprocessor', preprocessor),

              ('Huber', HuberRegressor()) ])  

# Lasso

pipe_Lasso = Pipeline(

    steps  = [ ('preprocessor', preprocessor),

               ('Lasso', Lasso(random_state=5)) ])

# ElasticNet

pipe_ElaNet = Pipeline(

    steps   = [ ('preprocessor', preprocessor),

                ('ElaNet', ElasticNet(random_state=5)) ])



# BayesianRidge

pipe_BayesRidge = Pipeline(

    steps   = [ ('preprocessor', preprocessor),

                ('BayesRidge', BayesianRidge(n_iter=500, compute_score=True)) ])

# GradientBoostingRegressor

pipe_GBR  = Pipeline(

    steps = [ ('preprocessor', preprocessor),

              ('GBR', GradientBoostingRegressor(random_state=5 )) ])



# XGBRegressor

pipe_XGB  = Pipeline(

    steps = [ ('preprocessor', preprocessor),

              ('XGB', XGBRegressor(objective='reg:squarederror', metric='rmse', 

                      random_state=5, nthread = -1)) ])

# LGBM

pipe_LGBM = Pipeline(

    steps= [('preprocessor', preprocessor),

            ('LGBM', LGBMRegressor(objective='regression', metric='rmse',

                                  random_state=5)) ])

# AdaBoostRegressor

pipe_ADA = Pipeline(

    steps= [('preprocessor', preprocessor),

            ('ADA', AdaBoostRegressor(DecisionTreeRegressor(), 

                random_state=5, loss='exponential')) ])
list_pipelines = [pipe_Linear, pipe_Ridge, pipe_Huber, pipe_Lasso, pipe_ElaNet]
print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")

print("-+"*30)

for pipe in list_pipelines :

    

    scores = cross_val_score(pipe, X_1, y_1, scoring='neg_mean_squared_error', cv=5)

    scores = np.sqrt(-scores)

    print(pipe.steps[1][0], "\t", 

          '{:08.6f}'.format(np.mean(scores)), "\t",  

          '{:08.6f}'.format(np.std(scores)),  "\t", 

          '{:08.6f}'.format(np.min(scores)))
list_pipelines = [pipe_GBR, pipe_XGB, pipe_LGBM, pipe_ADA]
print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")

print("-+"*30)



for pipe in list_pipelines :

    with warnings.catch_warnings():

        warnings.filterwarnings("ignore",category=FutureWarning)

        scores = cross_val_score(pipe, X_1, y_1, scoring='neg_mean_squared_error', cv=5)

        scores = np.sqrt(-scores)

        print(pipe.steps[1][0], "\t", 

          '{:08.6f}'.format(np.mean(scores)), "\t",  

          '{:08.6f}'.format(np.std(scores)),  "\t", 

          '{:08.6f}'.format(np.min(scores)))
list_scalers = [StandardScaler(), 

                RobustScaler(), 

                QuantileTransformer(output_distribution='normal')]
list_scalers = [StandardScaler()]
parameters_Linear = { 'preprocessor__num__scaler': list_scalers,

                     'Linear__fit_intercept':  [True,False],

                     'Linear__normalize':  [True,False] }



gscv_Linear = GridSearchCV(pipe_Linear, parameters_Linear, n_jobs=-1, 

                          scoring='neg_mean_squared_error', verbose=0, cv=5)

gscv_Linear.fit(X_1, y_1)
print(np.sqrt(-gscv_Linear.best_score_))  

gscv_Linear.best_params_
parameters_Ridge = { 'preprocessor__num__scaler': list_scalers,

                     'Ridge__alpha': [7,8,9],

                     'Ridge__fit_intercept':  [True,False],

                     'Ridge__normalize':  [True,False] }



gscv_Ridge = GridSearchCV(pipe_Ridge, parameters_Ridge, n_jobs=-1, 

                          scoring='neg_mean_squared_error', verbose=0, cv=5)

gscv_Ridge.fit(X_1, y_1)
print(np.sqrt(-gscv_Ridge.best_score_))  

gscv_Ridge.best_params_
parameters_Huber = { 'preprocessor__num__scaler': list_scalers,                   

                     'Huber__epsilon': [1.3, 1.35, 1.4],    

                     'Huber__max_iter': [150, 200, 250],                    

                     'Huber__alpha': [0.0005, 0.001, 0.002],

                     'Huber__fit_intercept':  [True], }



gscv_Huber = GridSearchCV(pipe_Huber, parameters_Huber, n_jobs=-1, 

                          scoring='neg_mean_squared_error', verbose=1, cv=5)

gscv_Huber.fit(X_1, y_1)
print(np.sqrt(-gscv_Huber.best_score_))  

gscv_Huber.best_params_
parameters_Lasso = { 'preprocessor__num__scaler': list_scalers,

                     'Lasso__alpha': [0.0005, 0.001],

                     'Lasso__fit_intercept':  [True],

                     'Lasso__normalize':  [True,False] }



gscv_Lasso = GridSearchCV(pipe_Lasso, parameters_Lasso, n_jobs=-1, 

                          scoring='neg_mean_squared_error', verbose=1, cv=5)

gscv_Lasso.fit(X_1, y_1)
print(np.sqrt(-gscv_Lasso.best_score_))  

gscv_Lasso.best_params_
parameters_ElaNet = { 'ElaNet__alpha': [0.0005, 0.001],

                      'ElaNet__l1_ratio':  [0.85, 0.9],

                      'ElaNet__normalize':  [True,False] }



gscv_ElaNet = GridSearchCV(pipe_ElaNet, parameters_ElaNet, n_jobs=-1, 

                          scoring='neg_mean_squared_error', verbose=1, cv=5)

gscv_ElaNet.fit(X_1, y_1)
print(np.sqrt(-gscv_ElaNet.best_score_))  

gscv_ElaNet.best_params_
list_pipelines_gscv = [gscv_Linear,gscv_Ridge,gscv_Huber,gscv_Lasso,gscv_ElaNet]
print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")

print("-+"*30)

for gscv in list_pipelines_gscv :

    

    scores = cross_val_score(gscv.best_estimator_, X_1, y_1, 

                             scoring='neg_mean_squared_error', cv=5)

    scores = np.sqrt(-scores)

    print(gscv.estimator.steps[1][0], "\t", 

          '{:08.6f}'.format(np.mean(scores)), "\t",  

          '{:08.6f}'.format(np.std(scores)),  "\t", 

          '{:08.6f}'.format(np.min(scores)))
parameters_GBR = { 'GBR__n_estimators':  [400], 

                   'GBR__max_depth':  [3,4],

                   'GBR__min_samples_leaf':  [5,6],                 

                   'GBR__max_features':  ["auto",0.5,0.7],                  

                 }

                   

gscv_GBR = GridSearchCV(pipe_GBR, parameters_GBR, n_jobs=-1, 

                        scoring='neg_mean_squared_error', verbose=1, cv=5)

gscv_GBR.fit(X_1, y_1)
print(np.sqrt(-gscv_GBR.best_score_))  

gscv_GBR.best_params_
parameters_XGB = { 'XGB__learning_rate': [0.021,0.022],

                   'XGB__max_depth':  [2,3],

                   'XGB__n_estimators':  [2000], 

                   'XGB__reg_lambda':  [1.5, 1.6], 

                   'XGB__reg_alpha':  [1,1.5],                   

# colsample_bytree , subsample               

                  }

                   

gscv_XGB = GridSearchCV(pipe_XGB, parameters_XGB, n_jobs=-1, 

                        scoring='neg_mean_squared_error', verbose=1, cv=5)

gscv_XGB.fit(X_1, y_1)
print(np.sqrt(-gscv_XGB.best_score_))  

gscv_XGB.best_params_
parameters_LGBM = { 'LGBM__learning_rate': [0.01,0.02],

                    'LGBM__n_estimators':  [1000], 

                    'LGBM__num_leaves':  [8,10],

                    'LGBM__bagging_fraction':  [0.7,0.8],

                    'LGBM__bagging_freq':  [1,2],                  

                   }



gscv_LGBM = GridSearchCV(pipe_LGBM, parameters_LGBM, n_jobs=-1, 

                       scoring='neg_mean_squared_error', verbose=1, cv=5)

gscv_LGBM.fit(X_1, y_1)
print(np.sqrt(-gscv_LGBM.best_score_))  

gscv_LGBM.best_params_
parameters_ADA = { 'ADA__learning_rate': [3.5],

                   'ADA__n_estimators':  [500], 

                   'ADA__base_estimator__max_depth':  [8,9,10],                  

                 }



pipe_ADA = Pipeline(

    steps= [('preprocessor', preprocessor),

            ('ADA', AdaBoostRegressor(

                DecisionTreeRegressor(min_samples_leaf=5,

                                      min_samples_split=5), 

                random_state=5,loss='exponential')) ])



gscv_ADA = GridSearchCV(pipe_ADA, parameters_ADA, n_jobs=-1, 

                       scoring='neg_mean_squared_error', verbose=1, cv=5)

gscv_ADA.fit(X_1, y_1)
print(np.sqrt(-gscv_ADA.best_score_))  

gscv_ADA.best_params_
list_pipelines_gscv = [gscv_GBR, gscv_XGB,gscv_LGBM,gscv_ADA]
print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")

print("-+"*30)

for gscv in list_pipelines_gscv :

    with warnings.catch_warnings():

        warnings.filterwarnings("ignore",category=FutureWarning)    

        scores = cross_val_score(gscv.best_estimator_, X_1, y_1, 

                             scoring='neg_mean_squared_error', cv=5)

        scores = np.sqrt(-scores)

        print(gscv.estimator.steps[1][0], "\t", 

          '{:08.6f}'.format(np.mean(scores)), "\t",  

          '{:08.6f}'.format(np.std(scores)),  "\t", 

          '{:08.6f}'.format(np.min(scores)))
linear_models = [gscv_Linear,gscv_Ridge,gscv_Huber,gscv_Lasso,gscv_ElaNet]

boost_models  = [gscv_GBR, gscv_XGB,gscv_LGBM,gscv_ADA]
pred_Linear = gscv_Linear.predict(df_test)

pred_Ridge  = gscv_Ridge.predict(df_test)

pred_Huber  = gscv_Huber.predict(df_test)

pred_Lasso  = gscv_Lasso.predict(df_test)

pred_ElaNet = gscv_ElaNet.predict(df_test)
predictions_linear = {'Linear': pred_Linear, 'Ridge': pred_Ridge, 'Huber': pred_Huber,

                      'Lasso':  pred_Lasso, 'ElaNet': pred_ElaNet }
for model,values in predictions_linear.items():

    str_filename = model + ".csv"

    print("witing submission to : ", str_filename)

    subm = pd.DataFrame()

    subm['Id'] = id_test

    subm['SalePrice'] = np.expm1(values)

    subm.to_csv(str_filename, index=False)
pred_Blend_1 = (pred_Lasso + pred_Ridge) / 2

sub_Blend_1 = pd.DataFrame()

sub_Blend_1['Id'] = id_test

sub_Blend_1['SalePrice'] = np.expm1(pred_Blend_1)

sub_Blend_1.to_csv('Blend_Ridge_Lasso.csv',index=False)

sub_Blend_1.head()
pred_Blend_2 = (pred_Lasso + pred_ElaNet) / 2

sub_Blend_2 = pd.DataFrame()

sub_Blend_2['Id'] = id_test

sub_Blend_2['SalePrice'] = np.expm1(pred_Blend_2)

sub_Blend_2.to_csv('Blend_2.csv',index=False)

sub_Blend_2.head()
pred_Blend_3 = (pred_Ridge + pred_Lasso + pred_ElaNet) / 3

sub_Blend_3 = pd.DataFrame()

sub_Blend_3['Id'] = id_test

sub_Blend_3['SalePrice'] = np.expm1(pred_Blend_3)

sub_Blend_3.to_csv('Blend_3.csv',index=False)

sub_Blend_3.head()
boost_models  = [gscv_GBR, gscv_XGB,gscv_LGBM,gscv_ADA]
pred_GBR  = gscv_GBR.predict(df_test)

pred_XGB  = gscv_XGB.predict(df_test)

pred_LGBM = gscv_LGBM.predict(df_test)

pred_ADA  = gscv_ADA.predict(df_test)
predictions_boost = {'GBR': pred_GBR, 'XGB': pred_XGB, 'LGBM': pred_LGBM,

                     'ADA': pred_ADA }
for model,values in predictions_boost.items():

    str_filename = model + ".csv"

    print("witing submission to : ", str_filename)

    subm = pd.DataFrame()

    subm['Id'] = id_test

    subm['SalePrice'] = np.expm1(values)

    subm.to_csv(str_filename, index=False)
predictions = {'Ridge': pred_Ridge, 'Lasso': pred_Lasso, 'ElaNet': pred_ElaNet, 

               'GBR': pred_GBR, 'XGB': pred_XGB, 'LGBM': pred_LGBM, 'ADA': pred_ADA}

df_predictions = pd.DataFrame(data=predictions) 

df_predictions.corr()
pred_Blend_10 = (pred_Ridge + pred_XGB) / 2

sub_Blend_10 = pd.DataFrame()

sub_Blend_10['Id'] = id_test

sub_Blend_10['SalePrice'] = np.expm1(pred_Blend_10)

sub_Blend_10.to_csv('Blend_Ridge_XGB.csv',index=False)

sub_Blend_10.head()
lnr = LinearRegression(n_jobs = -1)



rdg = Ridge(alpha=3.0, copy_X=True, fit_intercept=True, random_state=1)



rft = RandomForestRegressor(n_estimators = 12, max_depth = 3, n_jobs = -1, random_state=1)



gbr = GradientBoostingRegressor(n_estimators = 40, max_depth = 2, random_state=1)



mlp = MLPRegressor(hidden_layer_sizes = (90, 90), alpha = 2.75, random_state=1)
stack1 = StackingRegressor(regressors = [rdg, rft, gbr], 

                           meta_regressor = lnr)
pipe_STACK_1 = Pipeline(steps=[ ('preprocessor', preprocessor),

                                ('stack1', stack1) ])



pipe_STACK_1.fit(X_1, y_1)
pred_stack1 = pipe_STACK_1.predict(df_test)

sub_stack1 = pd.DataFrame()

sub_stack1['Id'] = id_test

sub_stack1['SalePrice'] = np.expm1(pred_stack1)

sub_stack1.to_csv('pipe_stack1.csv',index=False)
sub_stack1.head(10)