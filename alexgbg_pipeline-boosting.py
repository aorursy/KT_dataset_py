# LIBRARIES

# general purpose

import math

import numpy as np

import pandas as pd 

import plotly.graph_objects as go

import os



# data processing

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import QuantileTransformer

from scipy.special import boxcox1p



# modelling

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PowerTransformer

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split



# parameter tuning

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint 

from scipy.stats import gamma
# data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()
target = ['SalePrice']

feat = [var for var in train.columns if var not in ['Id'] + target]

catfeat = [var for var in feat

          if train[var].dtypes == 'object']

numfeat = [var for var in feat

          if train[var].dtypes in ['int64', 'float']]
fig = go.Figure()

for var in numfeat+target: 

    fig.add_trace(go.Box(y=train[var], 

                        name=var))

fig.show(renderer="notebook")
# Correlation Matrix

fig = go.Figure(data=go.Heatmap(z=train[numfeat+target].corr(),

                                x=train[numfeat+target].columns, 

                                y=train[numfeat+target].columns,

                                zmin=-1,

                                zmax=1,

                                colorscale="RdBu"))

fig.show(renderer="notebook")
# Scatter Plots

if True: 

    xname = numfeat[0]

    yname = "SalePrice"

    fig = go.Figure()

    for feat in numfeat: 

        fig.add_trace(go.Scatter(x=train[feat], 

                                 y=train[yname], 

                                 name=feat, 

                                mode="markers"))



    fig.show(renderer="notebook")
# set vars

missing_catfeat = {var for var in catfeat

                   if train[var].isnull().sum() !=0}

missing_numfeat = {var for var in numfeat

                  if train[var].isnull().sum() != 0}

missing_feat = missing_catfeat | missing_numfeat



# plot

missing = train[missing_feat].isnull().sum() / len(train) * 100

missing = missing.sort_values(ascending=False)

fig = go.Figure(data=go.Bar(y=missing, x=missing.index), 

               layout=go.Layout(yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="percentage missing"))))

fig.show(renderer="notebook")
# Data prep configuration stored in CSV

config = pd.read_csv("../input/config-final/config_final.csv")

config = config.fillna('None')

config.head()
# Remove outliers detected through visual inspection

train.drop(train[(train['GrLivArea']>4000) & 

                 (train['SalePrice']<300000)].index,

           inplace = True)
# config parsing

transformers = list()

for i in range(0,config.shape[0]): 

    steps = list()

    

    for step in ['imputation', 'transformation', 'encoding']: 

        if config[step][i] != 'None': 

            param  = config[step + "_param"][i].split(";")

            keys   = list(map(lambda x: x.split("=")[0].strip(), param))

            values = list(map(lambda x: eval(x.split("=")[1]),   param))

            kwargs = dict(zip(keys, values))

            steps.append((step, globals()[config[step][i]](**kwargs)))  

    

    transformers.append((config["name"][i], Pipeline(steps=steps), [config["name"][i]]))



# processing wrapper

preprocessor = ColumnTransformer(transformers=transformers)
y = [math.log(x) for x in train['SalePrice']]

X = np.ones((len(y),1))

baseline = Pipeline(steps=[('regression', LinearRegression())])

scores = -1 * cross_val_score(baseline, X, y, cv=5, scoring='neg_mean_squared_error')

print(np.mean([math.sqrt(x) for x in scores]))
# parameters

X =  train.drop(target, axis=1)

y = [math.log(x) for x in train['SalePrice']]

bt = Pipeline(steps=[

    ('dataprep', preprocessor),

    ('regression', GradientBoostingRegressor(learning_rate=0.1,

                                             n_estimators=380,

                                             min_samples_split=23, 

                                             min_samples_leaf=15,

                                             max_features=40,

                                             max_depth=3, 

                                             subsample=0.8

                                             ))])

param_dist = {"regression__n_estimators": randint(550, 750),

              "regression__learning_rate": gamma(a=4, scale=0.005)}

n_iter_search = 20



# random search

random_search = RandomizedSearchCV(bt, 

                                   param_distributions=param_dist, 

                                   n_iter=n_iter_search, 

                                   cv=5, 

                                   iid=False)

random_search.fit(X,y)

random_search.best_estimator_
# eval

scores = -1 * cross_val_score(random_search.best_estimator_, X, y, cv=5, scoring='neg_mean_squared_error')

print(np.mean([math.sqrt(x) for x in scores]))


random_search.best_estimator_.fit(train.drop('SalePrice', axis=1),y)

pred = list(map(lambda x: math.exp(x), random_search.best_estimator_.predict(test)))

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': pred})

submission.to_csv('submission2.csv', index=False)

submission.head()