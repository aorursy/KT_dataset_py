import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt





df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')





#training data treatment

print(df.info())



#Removing columns having more than 30% null values

mask_null = df.isnull().sum() / len(df) < 0.3

df = df.loc[: , mask_null]



#Information of all non-numric data

#converting some columns to their true data type for better data handling

#columns belong to 'category' dtype

for i  in ['MSSubClass', 'OverallQual', 'OverallCond']:

    df[i] = df[i].astype('category')



#columns belong to 'datetime'

for i in ['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt', 'MoSold']:

    if i == 'MoSold':

        df[i] = pd.to_datetime(df[i], format= '%m')

    else :

        df[i] = pd.to_datetime(df[i])



#Columns with the frequency of top category

print('\n', df.describe(exclude= 'number').loc['freq', :].sort_values())

sns.relplot(x= 'Street', y= 'SalePrice', data= df, palette= 'husl')

sns.relplot(x= 'Condition2', y= 'SalePrice', data= df, palette= 'husl')

sns.relplot(x= 'RoofMatl', y= 'SalePrice', data= df, palette= 'husl')

plt.xticks(rotation= 90)

sns.relplot(x= 'Heating', y= 'SalePrice', data= df, palette= 'husl')
#Drop categorical features with less information

df.drop(['Utilities', 'Street', 'Condition2', 'RoofMatl', 'Heating', 'YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt', 'MoSold'], axis= 1, inplace= True)
#import required functions

from sklearn.linear_model import LassoCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#for recursice feature selection

from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler



#Get numeric features

df_num = df[df.describe().columns]



#drop null as sklearn doesn't work with them

df_num.dropna(inplace= True)



#Scaling data for LassoCV

scaler = StandardScaler()

std_data = scaler.fit_transform(df_num.drop(['SalePrice'], axis= 1))



#Linear regressor

lr = LassoCV()

lr.fit(std_data, df_num['SalePrice'])

mask_lr = lr.coef_ != 0



#Random Forest 

rf = RandomForestRegressor()

rfe_rf = RFE(estimator = rf, n_features_to_select = 10, step = 5)

rfe_rf.fit(df_num.drop(['SalePrice'], axis = 1), df_num['SalePrice'])

mask_rf = rfe_rf.support_



#Gradient boosting

rfe_gb = RFE(estimator = GradientBoostingRegressor(), n_features_to_select = 10, step = 5)

rfe_gb.fit(df_num.drop(['SalePrice'], axis= 1), df_num['SalePrice'])

mask_gb = rfe_gb.support_



mask_final = np.sum([mask_lr, mask_rf, mask_gb], axis= 0) > 2





X = df_num.drop(['SalePrice'], axis= 1)

#Final numeric columns to include

columns_to_include = X.loc[: , mask_final].columns

columns_to_exclude = list(set(df.describe().columns) - set(columns_to_include))

columns_to_exclude.remove('SalePrice')

#Changing the original dataframe 

df.drop(columns_to_exclude, axis= 1, inplace = True)

print(df.shape)
print(df.isnull().sum())
df.info()



#Separate numeric and categorical data

numeric_features = df.drop('SalePrice', axis= 1).describe().columns.tolist()





non_numeric_features = df.describe(exclude= 'number').columns.tolist()



#Import required libraries

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.model_selection import  RandomizedSearchCV

from sklearn.preprocessing import  StandardScaler, FunctionTransformer

from sklearn.impute import SimpleImputer

from sklearn.feature_extraction import DictVectorizer

from sklearn_pandas import DataFrameMapper, CategoricalImputer

import xgboost as xgb



#Apply numeric imputer

numeric_imputer_mapper = DataFrameMapper([

    ([num_features], SimpleImputer(strategy= 'median')) for num_features in numeric_features

                                        ],

    input_df= True,

    df_out= True

)



#Apply imputer for categorical data

cat_imputer_mapper = DataFrameMapper([

    (cat_features, CategoricalImputer()) for cat_features in non_numeric_features

                                    ],

    input_df= True,

    df_out= True

)



print(cat_imputer_mapper.fit_transform(df).head())



#Make separate pipelines for both types of columns

numeric_pipeline = Pipeline([

    ('imputer', numeric_imputer_mapper),

    ('scaler', StandardScaler())

])



#prepare inpute for dictvectorizer

Dictifier = FunctionTransformer(lambda x: x.to_dict('record'))

    

cat_pipeline = Pipeline([

    ('imputer', cat_imputer_mapper),

    ('dictifier', Dictifier),

    ('vectorizer', DictVectorizer(sort= False))

])



#Compine two pipelines for xgboost

union = FeatureUnion([

    ('numeric_pipe', numeric_pipeline),

    ('cat_pipe', cat_pipeline)

])



#Final pipeline

xgb_pipeline = Pipeline([

    ('union', union),

    ('xgb_model', xgb.XGBRegressor())

])
#Create a parameter grid for xg boost

xgb_params_grid = {

    'xgb_model__learning_rate': np.arange(0.05, 1, 0.05),

    'xgb_model__max_depth': np.arange(2, 20, 1),

    'xgb_model__subsample': np.arange(0.05, 1, 0.05),

    'xgb_model__colsample_bytree': np.arange(.1,1,.05)

}



cv_result = RandomizedSearchCV(estimator= xgb_pipeline, 

                              param_distributions = xgb_params_grid,

                              n_iter= 10,

                              scoring= 'neg_mean_squared_error',

                              cv= 4, n_jobs= -1)

#Make feature and target data

X, y= df.drop('SalePrice', axis= 1), df['SalePrice']



#Fit the model

cv_result.fit(X, y)



#Check efficiency

print('Best RMSE: ', np.sqrt(np.abs(cv_result.best_score_)))
#Load test data

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#Inspect the data

#print(df_test.info())



#Extract the features we need

df_test_final = df_test.loc[: , X.columns]



prediction = cv_result.predict(df_test_final)



submission = pd.DataFrame({

    'Id': df_test['Id'],

    'SalePrice': prediction

})







submission.set_index('Id', inplace= True)

print(submission.head())



submission.to_csv('submission_house_data.csv')