import os

import numpy as np 

import pandas as pd 



# For data visualization

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

%matplotlib inline



# For data modeling & prediction

from scipy import stats

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor



from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor

from sklearn.linear_model import BayesianRidge, LinearRegression

import xgboost as xgb

from sklearn.metrics import mean_absolute_error



from sklearn.model_selection import cross_validate, train_test_split

from statistics import mean

from sklearn.model_selection import KFold



import warnings

warnings.filterwarnings("ignore") 
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip", compression='zip')

df_features = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip", compression='zip')

df_test = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip", compression='zip')

df_stores = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")
df_train.tail(3)
df_test.tail(3)
df_stores.tail(3)
df_features.head(3)
df_train = pd.merge(df_train,df_features, on = ['Store','Date','IsHoliday'],how='inner')

df_train = pd.merge(df_train,df_stores, on= 'Store',how='inner')

df_test = pd.merge(df_test,df_features, on = ['Store','Date','IsHoliday'],how='inner')

df_test = pd.merge(df_test,df_stores, on= 'Store',how='inner')
"train", df_train.isna().mean(),"---------------------", 
def convert_dates(dataframe):

    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    dataframe['year'] = dataframe['Date'].dt.year

    dataframe['week'] = dataframe.Date.dt.week 

    

    return dataframe



df_test = convert_dates(df_test)

df_train = convert_dates(df_train)
to_categorical = ['Store', 'Dept', 'IsHoliday', 'Type', 'year', 'week']

for column in to_categorical:

    df_train[column] = df_train[column].astype('category')

    

df_train.dtypes
df_train[['Dept', 'Store', 'week', 'year', 'IsHoliday']] = df_train[['Dept', 'Store', 'week', 'year', 'IsHoliday']].astype('int')



plt.figure(figsize=(18,12))

corr = df_train.corr()

np.fill_diagonal(corr.values, np.nan)



sns.heatmap(corr, annot=True, fmt='.2f')
def boxplot(column, x_size=15, y_size=10):

    fig = plt.figure(figsize=(x_size,y_size))

    sns.boxplot(y=df_train.Weekly_Sales, x=df_train[column])

    plt.ylabel('Weekly_Sales')

    plt.xlabel(column)

boxplot('Store')
boxplot('Dept', x_size=25)
boxplot('week')
boxplot('year')
boxplot('Type')
boxplot('IsHoliday')
def correlation(column):

    print("----------------------------Column name: "+column+"----------------------------")

    print("Correlation: " + str(df_train['Weekly_Sales'].corr(df_train[column])))

    print("\n")
correlation("CPI")

correlation("Unemployment")

correlation("Temperature")

correlation("Size")

correlation("Fuel_Price")

correlation("Unemployment")

correlation("MarkDown1")

correlation("MarkDown2")

correlation("MarkDown3")

correlation("MarkDown4")

correlation("MarkDown5")
df_train.groupby('Store')['Size'].nunique()
df_train = pd.get_dummies(df_train, columns=['Type'])

df_train.columns
df_test = pd.get_dummies(df_test, columns=['Type'])

df_test.columns
df_train.columns
df_train.dtypes
mean_sales = df_train.groupby(["Store", "Dept", "week"], as_index=False).agg({"Weekly_Sales": "mean"})

df_val = df_test.merge(mean_sales, on=['Store', 'Dept', 'week'], how='left')

sample_submission = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip")



test_ids = df_test.Store.astype(str) + '_' + df_test.Dept.astype(str) + '_' + df_test.Date.astype(str)

sample_submission['Id'] = test_ids.values

sample_submission["Weekly_Sales"] = df_val["Weekly_Sales"]



# apparently there are some missing values. I will fill the NaN values with 0 (I know I miss some score :( ).

sample_submission = sample_submission.fillna(0)

sample_submission.to_csv('submission_simple_mean.csv',index=False)
kfold = KFold(n_splits=5, random_state=35)



x = df_train.loc[:, df_train.columns != 'Weekly_Sales']

x = x.drop(['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5', 'CPI', 

            'Unemployment', 'Size', 'IsHoliday', 'Type_A', 'Type_B', 'Type_C', 'year', 'Date'], axis=1)



y = df_train.loc[:, df_train.columns == 'Weekly_Sales']



models = {

    

    'xgboost' : xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10),

    'Bayesian' : BayesianRidge(),

    'LinearRegression': LinearRegression(),

    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=1),

    'AdaBoostRegressor' : AdaBoostRegressor(n_estimators=50, learning_rate=.1, loss='square'),

    'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=50, max_features='auto', random_state=35),

    'RandomForestRegressor': RandomForestRegressor(n_estimators=50, random_state=35),

}



for model_name, model in models.items():

    results = cross_validate(model, x,y , cv=kfold, scoring=['neg_mean_absolute_error'], return_estimator=False)

    print(model_name, mean(results['test_neg_mean_absolute_error']), mean(results['fit_time']), mean(results['score_time']))
x = df_train.loc[:, df_train.columns != 'Weekly_Sales']

x = x.drop(['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5', 'CPI', 

            'Unemployment', 'Date'], axis=1)



y = df_train.loc[:, df_train.columns == 'Weekly_Sales']



models = {

    

    'xgboost' : xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10),

    'Bayesian' : BayesianRidge(),

    'LinearRegression': LinearRegression(),

    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=1),

    'AdaBoostRegressor' : AdaBoostRegressor(n_estimators=50, learning_rate=.1, loss='square'),

    'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=50, max_features='auto', random_state=35),

    'RandomForestRegressor': RandomForestRegressor(n_estimators=50, random_state=35),

}



res = {}



for model_name, model in models.items():

    results = cross_validate(model, x,y , cv=kfold, scoring=['neg_mean_absolute_error'], return_estimator=False)

    print(model_name, mean(results['test_neg_mean_absolute_error']), mean(results['fit_time']), mean(results['score_time']))
x = df_train.loc[:, df_train.columns != 'Weekly_Sales']

x = x.drop(['Date'], axis=1)

x = x.fillna(0)

y = df_train.loc[:, df_train.columns == 'Weekly_Sales']



models = {

    'xgboost' : xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10),

    'Bayesian' : BayesianRidge(),

    'LinearRegression': LinearRegression(),

    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=1),

    'AdaBoostRegressor' : AdaBoostRegressor(n_estimators=50, learning_rate=.1, loss='square'),

    'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=50, max_features='auto', random_state=35),

    'RandomForestRegressor': RandomForestRegressor(n_estimators=50, random_state=35),

}



res = {}



for model_name, model in models.items():

    results = cross_validate(model, x,y , cv=kfold, scoring=['neg_mean_absolute_error'], return_estimator=False)

    print(model_name, mean(results['test_neg_mean_absolute_error']), mean(results['fit_time']), mean(results['score_time']))
x = df_train.loc[:, df_train.columns != 'Weekly_Sales']

x = x.drop(['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5', 'CPI', 

            'Unemployment', 'Date'], axis=1)



y = df_train.loc[:, df_train.columns == 'Weekly_Sales']



extratreeregressor = ExtraTreesRegressor(n_estimators=50, max_features='auto', random_state=35)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=35)

extratreeregressor.fit(X_train, y_train)

y_pred = extratreeregressor.predict(X_test)

mean_absolute_error(y_pred, y_test)
x_val = df_test.drop(['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5', 'CPI', 

            'Unemployment', 'Date'], axis=1)



y_val = extratreeregressor.predict(x_val)
test_ids = df_test.Store.astype(str) + '_' + df_test.Dept.astype(str) + '_' + df_test.Date.astype(str)

sample_submission['Id'] = test_ids.values

sample_submission["Weekly_Sales"] = y_val



sample_submission = sample_submission.fillna(0)

sample_submission.to_csv('submission_extratreeregressor.csv',index=False)
from sklearn.decomposition import PCA



x = df_train.loc[:, df_train.columns != 'Weekly_Sales']

x = x.drop(['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5', 'CPI', 

            'Unemployment', 'Date'], axis=1)

x = x.fillna(0)

y = df_train.loc[:, df_train.columns == 'Weekly_Sales']



pca = PCA(n_components=5)

pca.fit(x)

pca_features = pca.transform(x)



columns = ['pca_%i' % i for i in range(5)]

x = pd.DataFrame(pca_features, columns=columns, index=x.index)
extratreeregressor_pca = ExtraTreesRegressor(n_estimators=50, max_features='auto', random_state=35)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=35)

extratreeregressor_pca.fit(X_train, y_train)

y_pred = extratreeregressor_pca.predict(X_test)

mean_absolute_error(y_pred, y_test)
x_val = df_test.drop(['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5', 'CPI', 

            'Unemployment', 'Date'], axis=1)

pca_features_val = pca.transform(x_val)

columns = ['pca_%i' % i for i in range(5)]



x_val = pd.DataFrame(pca_features_val, columns=columns, index=x_val.index)



y_val = extratreeregressor_pca.predict(x_val)
test_ids = df_test.Store.astype(str) + '_' + df_test.Dept.astype(str) + '_' + df_test.Date.astype(str)

sample_submission['Id'] = test_ids.values

sample_submission["Weekly_Sales"] = y_val



sample_submission = sample_submission.fillna(0)

sample_submission.to_csv('submission_extratreeregressor_pca.csv',index=False)
x = df_train.loc[:, df_train.columns != 'Weekly_Sales']

x = pd.concat([x, pd.DataFrame(pca_features)], axis=1)



x = x.drop(['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5', 'CPI', 

            'Unemployment', 'Date'], axis=1)

x = x.fillna(0)

y = df_train.loc[:, df_train.columns == 'Weekly_Sales']
extratreeregressor_pca_allfeatures = ExtraTreesRegressor(n_estimators=50, max_features='auto', random_state=35)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=35)

extratreeregressor_pca_allfeatures.fit(X_train, y_train)

y_pred = extratreeregressor_pca_allfeatures.predict(X_test)

mean_absolute_error(y_pred, y_test)
x_val = df_test.drop(['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5', 'CPI', 

            'Unemployment', 'Date'], axis=1)

pca_features_val = pca.transform(x_val)

x_val = pd.concat([x_val, pd.DataFrame(pca_features_val)], axis=1)



y_val = extratreeregressor_pca_allfeatures.predict(x_val)
test_ids = df_test.Store.astype(str) + '_' + df_test.Dept.astype(str) + '_' + df_test.Date.astype(str)

sample_submission['Id'] = test_ids.values

sample_submission["Weekly_Sales"] = y_val



sample_submission = sample_submission.fillna(0)

sample_submission.to_csv('submission_extratreeregressor_pca_allfeatures.csv',index=False)