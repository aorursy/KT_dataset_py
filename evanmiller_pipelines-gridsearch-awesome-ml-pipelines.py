import numpy as np

import pandas as pd

import lightgbm as lgb

from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest, VarianceThreshold

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import Imputer, PolynomialFeatures, StandardScaler, OneHotEncoder, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor

import os

import time



import warnings

warnings.filterwarnings("ignore")



def MAE(y, ypred):

    

    import numpy as np

    

    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)   
train = pd.read_csv("../input/train_2016_v2.csv")

properties = pd.read_csv('../input/properties_2016.csv')



for c, dtype in zip(properties.columns, properties.dtypes):	

    if dtype == np.float64:

        properties[c] = properties[c].astype(np.float32)



df_train = (train.merge(properties, how='left', on='parcelid')

            .drop(['parcelid', 'transactiondate', 'propertyzoningdesc', 

                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1))



train_columns = df_train.columns 
valid = df_train.iloc[1:20000, :]

train = df_train.iloc[20001:90275, :]



y_train = train['logerror'].values

y_valid = valid['logerror'].values



x_train = train.drop('logerror', axis = 1)

x_valid = valid.drop('logerror', axis = 1)



idVars = [i for e in ['id',  'flag', 'has'] for i in list(train_columns) if e in i] + ['fips', 'hashottuborspa']

countVars = [i for e in ['cnt',  'year', 'nbr', 'number'] for i in list(train_columns) if e in i]

taxVars = [col for col in train_columns if 'tax' in col and 'flag' not in col]

          

ttlVars = idVars + countVars + taxVars

dropVars = [i for e in ['census',  'tude', 'error'] for i in list(train_columns) if e in i]

contVars = [col for col in train_columns if col not in ttlVars + dropVars]



for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)

    

for c in x_valid.dtypes[x_valid.dtypes == object].index.values:

    x_valid[c] = (x_valid[c] == True)   
print(contVars)



x_train_cont = x_train[contVars]

x_valid_cont = x_valid[contVars]
pipeline = Pipeline(

                    [('imp', Imputer(missing_values='NaN', strategy = 'median', axis=0)),

                     ('feat_select', SelectKBest(k = 5)),

                     ('lgbm', LGBMRegressor())

                     

])



pipeline.fit(x_train_cont, y_train)   



y_pred = pipeline.predict(x_valid_cont)

print('MAE on validation set: %s' % (round(MAE(y_valid, y_pred), 5)))
pipeline = Pipeline(

                    [('imp', Imputer(missing_values='NaN', axis=0)),

                     ('feat_select', SelectKBest()),

                     ('lgbm', LGBMRegressor())

                     

])



parameters = {}

parameters['imp__strategy'] = ['mean', 'median', 'most_frequent']

parameters['feat_select__k'] = [5, 10]



CV = GridSearchCV(pipeline, parameters, scoring = 'mean_absolute_error', n_jobs= 1)

CV.fit(x_train_cont, y_train)   



print('Best score and parameter combination = ')



print(CV.best_score_)    

print(CV.best_params_)    



y_pred = CV.predict(x_valid_cont)

print('MAE on validation set: %s' % (round(MAE(y_valid, y_pred), 5)))
from sklearn.base import BaseEstimator, TransformerMixin



class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, subset):

        self.subset = subset



    def transform(self, X, *_):

        return X.loc[:, self.subset]



    def fit(self, *_):

        return self
contExtractor = ColumnSelector(contVars)

x_train_cont_test = contExtractor.transform(x_train).head()



x_train_cont.head().equals(x_train_cont_test)
pipeline = Pipeline([

                    ('tax_dimension', ColumnSelector(taxVars)),

                    ('imp', Imputer(missing_values='NaN', axis=0)),

                    ('column_purge', SelectKBest()),

                    ('lgbm', LGBMRegressor())

                     

])



parameters = dict(imp__strategy=['mean', 'median', 'most_frequent'],

                    column_purge__k=[5, 2, 1] 



)   



CV = GridSearchCV(pipeline, parameters, scoring = 'neg_mean_absolute_error', n_jobs= 1)

CV.fit(x_train, y_train)   



print(CV.best_params_)    

print(CV.best_score_)    



y_pred = CV.predict(x_valid)

print('MAE on validation set: %s' % (round(MAE(y_valid, y_pred), 5)))
pipeline = Pipeline([

        

    ('unity', FeatureUnion(

        transformer_list=[



            ('cont_portal', Pipeline([

                ('selector', PortalToColDimension(contVars)),

                ('cont_imp', Imputer(missing_values='NaN', strategy = 'median', axis=0)),

                ('scaler', StandardScaler())             

            ])),

            ('tax_portal', Pipeline([

                ('selector', PortalToColDimension(taxVars)),

                ('tax_imp', Imputer(missing_values='NaN', strategy = 'most_frequent', axis=0)),

                ('scaler', MinMaxScaler(copy=True, feature_range=(0, 3)))

            ])),

        ],

    )),

    ('column_purge', SelectKBest(k = 5)),    

    ('lgbm', LGBMRegressor()),

])



parameters = {}

parameters['column_purge__k'] = [5, 10]



grid = GridSearchCV(pipeline, parameters, scoring = 'neg_mean_absolute_error', n_jobs= 2)

grid.fit(x_train, y_train)   



print('Best score and parameter combination = ')



print(grid.best_score_)    

print(grid.best_params_)    



y_pred = grid.predict(x_valid)

print('MAE on validation set: %s' % (round(MAE(y_valid, y_pred), 5)))
from sklearn.externals import joblib

joblib.dump(grid.best_estimator_, 'rick.pkl')