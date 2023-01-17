# This Python 3 environment comes with many helpful analytics libraries installed

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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import missingno as msno 

import seaborn as sns



from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer



from sklearn.model_selection import cross_val_score, GridSearchCV



from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from xgboost import XGBRegressor
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 20)
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.shape
test.shape
test.info()
train.info()
train.head()
test.head()
X_train = train.drop(['Id','SalePrice'], axis=1)
missingdata_df = X_train.columns[X_train.isnull().any()].tolist()# include column only with null values

msno.matrix(train[missingdata_df])
X_train[missingdata_df].isna().sum().sort_values(ascending=False)[:20]
msno.bar(X_train[missingdata_df], color="turquoise", figsize=(30,18))
msno.heatmap(X_train)
X_train.describe()
sns.distplot(train['SalePrice'],color = 'turquoise')
target = np.log(train['SalePrice'])

target.skew()

plt.hist(target,color='turquoise')
sns.boxplot(train['SalePrice'],color = 'turquoise')
sns.boxplot(target,color = 'turquoise')
train1 = train.drop(['Id','PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis = 1)
numeric_data = train1 .select_dtypes(include=[np.number])

categorical_data = train1 .select_dtypes(exclude=[np.number])



print(numeric_data.shape)

print(categorical_data.shape)

numeric_data.head()
# fill missing values with mean column values

numeric_data.fillna(numeric_data.mean(), inplace=True)
numeric_data.info()
# Compute the correlation matrix

corr = numeric_data.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

plt.title('Correlation of Numeric Features',y=1,size=16)



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
correlation = numeric_data.corr()

print(correlation['SalePrice'].sort_values(ascending = False),'\n')




k= 11 # just taking 10(excluding salesprice) column with largest correlation coefficients with sales price

cols = correlation.nlargest(k,'SalePrice')['SalePrice'].index

print(cols)

cm = np.corrcoef(train[cols].values.T)

f , ax = plt.subplots(figsize = (14,12))

plt.title('Correlation of 10 numerical features with highest R with Sales Price',y=1,size=16)

sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="black",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
sns.set()

columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt','YearRemodAdd']

sns.pairplot(train[columns],height = 2 ,kind ='scatter',diag_kind='kde')

plt.show()
train[columns].hist(bins=15, figsize=(70, 90), layout=(13, 3));
plt.hist(train1['1stFlrSF'],color='turquoise')

plt.hist(train1['GrLivArea'],color='turquoise')

for c in categorical_data:

    train1[c] = train1[c].astype('category')

    if train1[c].isnull().any():

        train1[c] = train1[c].cat.add_categories(['MISSING'])

        train1[c] = train1[c].fillna('MISSING')



def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)

f = pd.melt(train, id_vars=['SalePrice'], value_vars=categorical_data)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, height=5)

g = g.map(boxplot, "value", "SalePrice")
categorical_data.fillna('MISSING', inplace=True)
categorical_data.info()
from scipy import stats

from scipy.stats import kendalltau

for (columnName, columnData) in categorical_data.iteritems():

    coef, p = kendalltau(categorical_data[columnName],train['SalePrice'],initial_lexsort=None, nan_policy='propagate', method='auto')

    alpha = 0.005

    if p < alpha and coef >= 0.3 or coef <= -0.3:

        print('Samples are  correlated ,p=%.3f' % p,'Kendall correlation coefficient: %.3f' % coef, columnName)

    else:

        next

        

    
categorical_data_columns = ['ExterQual','Foundation'

,'BsmtQual','HeatingQC','KitchenQual','GarageType','GarageFinish']



categorical_data_df = categorical_data[categorical_data_columns]



encoder=OneHotEncoder(sparse=False,handle_unknown='ignore')



train_X_encoded = pd.DataFrame (encoder.fit_transform(categorical_data_df[['ExterQual','Foundation'

,'BsmtQual','HeatingQC','KitchenQual','GarageType','GarageFinish']]))



train_X_encoded.columns = encoder.get_feature_names(['ExterQual','Foundation'

,'BsmtQual','HeatingQC','KitchenQual','GarageType','GarageFinish'])



categorical_data_df.drop(['ExterQual','Foundation'

,'BsmtQual','HeatingQC','KitchenQual','GarageType','GarageFinish'] ,axis=1, inplace=True)



cat_data_df= pd.concat([categorical_data_df, train_X_encoded ], axis=1)
numeric_data['1stFlrSF'] = np.log(numeric_data['1stFlrSF'])

numeric_data['GrLivArea'] = np.log(numeric_data['GrLivArea'])
numerical_data_df = pd.DataFrame(numeric_data,columns =['1stFlrSF','GrLivArea','YearBuilt','YearRemodAdd','FullBath','OverallQual', 'GarageArea']) 
X_train = pd.concat([numerical_data_df, cat_data_df], axis=1)
X_train.head()
print(X_train. columns) 
y_train = np.log(train.SalePrice)
from sklearn.model_selection import cross_val_score
regressor = LinearRegression()  

regressor.fit(X_train, y_train)
scores = cross_val_score(

regressor, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

scores
print('Results by fold:\n', scores, '\n') 

print('Mean CV Score:', np.mean(scores))
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV



# Use grid search to tune the parameters:



parametersGrid = {"max_iter": [10,100,1000,10000],

                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],

                      "l1_ratio": [0, 0.25, 0.5, 0.75, 1.0]}



eNet = ElasticNet()

grid = GridSearchCV(eNet, parametersGrid, scoring='neg_root_mean_squared_error', cv=10 ,refit='True', verbose = 10, n_jobs=-1)

grid.fit(X_train, y_train)

print(grid.best_score_) 

print(grid.best_params_)
en_model = grid.best_estimator_ 

print('Number of Features Kept: ', np.sum(en_model.coef_ != 0)) 

print('Number of Features Dropped:', np.sum(en_model.coef_ == 0))
cv_results = cross_val_score(grid.best_estimator_, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')



print('Results by fold:\n', cv_results, '\n') 

print('Mean CV Score:', np.mean(cv_results))
from sklearn.ensemble import RandomForestRegressor

param_grid={'max_depth': [4,6,8,10],

            'n_estimators': [12,14,16,18]}



rf = RandomForestRegressor()







gsc = GridSearchCV(RandomForestRegressor(),param_grid, scoring='neg_root_mean_squared_error', cv=10 ,refit='True', verbose = 10, n_jobs=-1)

gsc.fit(X_train, y_train)

print(gsc.best_score_) 

print(gsc.best_params_)    

    
rf_cv_results = cross_val_score(gsc.best_estimator_, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')



print('Results by fold:\n', rf_cv_results, '\n') 

print('Mean CV Score:', np.mean(rf_cv_results))
rf_model = gsc.best_estimator_
param_grid = {'regressor__learning_rate' : [0.01,0.1, 0.5, 0.9],

    'regressor__alpha' : [0, 1, 10],

    'regressor__max_depth': [2,4, 6]}

XGD = XGBRegressor(n_estimators=300, subsample=0.5)

np.random.seed(1)

xgd_grid_search = GridSearchCV(XGD, param_grid, cv=10, scoring='neg_root_mean_squared_error',

                              refit='True', verbose = 10, n_jobs=-1)

xgd_grid_search.fit(X_train, y_train)



print(xgd_grid_search.best_score_)

print(xgd_grid_search.best_params_)
xgb_model = xgd_grid_search.best_estimator_
ensemble = VotingRegressor(

    estimators = [

        ('en_model' , grid.best_estimator_),

        ('rf_model' , gsc.best_estimator_),

        ('xgb_model', xgd_grid_search.best_estimator_)

    ]

)



cv_results = cross_val_score(ensemble, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')



print('Results by fold:\n', cv_results, '\n')

print('Mean CV Score:', np.mean(cv_results))
ensemble.fit(X_train, y_train)

ensemble.score(X_train, y_train)
X_test_num = test[['1stFlrSF', 'GrLivArea', 'YearBuilt', 'YearRemodAdd', 'FullBath',

       'OverallQual', 'GarageArea']]
X_test_num.head()
X_test_num.fillna(numeric_data.mean(), inplace=True)
X_test_num['1stFlrSF'] = np.log(X_test_num['1stFlrSF'])

X_test_num['GrLivArea'] = np.log(X_test_num['GrLivArea'])
X_test_cat = test[['ExterQual', 'Foundation', 'BsmtQual',

       'HeatingQC', 'KitchenQual', 'GarageType', 'GarageFinish']]
X_test_cat.fillna('MISSING', inplace=True)
test_X_encoded = pd.DataFrame (encoder.transform(X_test_cat[['ExterQual','Foundation'

,'BsmtQual','HeatingQC','KitchenQual','GarageType','GarageFinish']]))



test_X_encoded.columns = encoder.get_feature_names(['ExterQual','Foundation'

,'BsmtQual','HeatingQC','KitchenQual','GarageType','GarageFinish'])



X_test_cat.drop(['ExterQual','Foundation'

,'BsmtQual','HeatingQC','KitchenQual','GarageType','GarageFinish'] ,axis=1, inplace=True)



cat_data_df= pd.concat([X_test_cat, test_X_encoded ], axis=1)
X_test = pd.concat([cat_data_df, X_test_num], axis=1)
X_test.shape
submission = sample_submission.copy()

submission.SalePrice = np.exp(en_model.predict(X_test))



submission.to_csv('my_submission.csv', index=False)

submission.head()