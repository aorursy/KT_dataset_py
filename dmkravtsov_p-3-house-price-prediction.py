# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

from scipy.stats import norm, skew #for some statistics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading file and name it df

df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df.head()


df.plot.scatter(x='GrLivArea', y='SalePrice', figsize=(16, 6), ylim=(0,800000))

# Deleting outliers

df = df.drop(df[(df['GrLivArea']>4500) & (df['SalePrice']<300000)].index)



df.drop(df[df['TotalBsmtSF'] > 5000].index, inplace=True)



df.drop(df[df['1stFlrSF'] > 4000].index,inplace=True)





df['SalePrice'] = np.log(df['SalePrice'])

sns.distplot(df['SalePrice'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(df['SalePrice'], plot=plt)



df['GrLivArea'] = np.log(df['GrLivArea'])

sns.distplot(df['GrLivArea'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(df['GrLivArea'], plot=plt)

y = df['SalePrice']

# some feature engeneering



df['MSSubClass'] = df['MSSubClass'].apply(str)

df['YrSold'] = df['YrSold'].astype(str)

df['MoSold'] = df['MoSold'].astype(str)



df["Functional"] = df["Functional"].fillna("Typ")



df['YrBltAndRemod']=df['YearBuilt']+df['YearRemodAdd']

df['TotalSF']=df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']



df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +

                                 df['1stFlrSF'] + df['2ndFlrSF'])



df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +

                               df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))



df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] +

                              df['EnclosedPorch'] + df['ScreenPorch'] +

                              df['WoodDeckSF'])



'''

df.sort_values(by = 'GrLivArea', ascending = False)[:2]

df = df.drop(df[df['Id'] == 1299].index, inplace = True)

df = df.drop(df[df['Id'] == 524].index, inplace = True)

df = df.drop(df[df['Id'] == 30].index, inplace = True)

df = df.drop(df[df['Id'] == 88].index, inplace = True)

df = df.drop(df[df['Id'] == 462].index, inplace = True)

df = df.drop(df[df['Id'] == 631].index, inplace = True)

df = df.drop(df[df['Id'] == 1322].index, inplace = True)

'''
# checking numeric columns

numeric_columns = df.select_dtypes(include=[np.number])

numeric_columns
# checking for zeroes in numeric coils

(numeric_columns == 0).sum().sort_values(ascending = False).head(30)


# delete numeric features with zero-values more than 50%

numeric_columns = numeric_columns.loc[:, ((numeric_columns == 0).sum(axis=0) <= len(numeric_columns.index)*0.5)]

numeric_columns
#checking for NaNs in numeric features

numeric_columns.isnull().sum().sort_values(ascending = False).head()
# some feature engeneering with NaNs in numerical columns



numeric_columns['YearBuiltCut'] = pd.qcut(numeric_columns['YearBuilt'], 10)

numeric_columns['GarageYrBlt'] = numeric_columns.groupby(['YearBuiltCut'])['GarageYrBlt'].transform(lambda x : x.fillna(x.median()))

numeric_columns['GarageYrBlt'] = numeric_columns['GarageYrBlt'].astype(int)

numeric_columns.drop('YearBuiltCut', axis=1, inplace=True)



numeric_columns['LotFrontage'].fillna(numeric_columns['LotFrontage'].median(), inplace = True)

numeric_columns['LotFrontage'] = numeric_columns['LotFrontage'].astype(int)
# checking for Object columns

object_columns = df.select_dtypes(include=[np.object])

object_columns
# checking for NANs in objects



object_columns.isnull().sum().sort_values(ascending = False).head(17)
# removing objects with nan-values more than 50% in frame

object_columns = object_columns.loc[:, (object_columns.isnull().sum(axis=0) <= len(df.index)*0.5)]

# replace NaNs with MODE

for col in object_columns:

    object_columns[col].fillna('None', inplace = True)
# checking for zeros in df

#(df == 0).sum().sort_values(ascending = False).head(30)
# delete features with zero-values more than 50%

#df = df.loc[:, ((df == 0).sum(axis=0) <= len(df.index)*0.5)]
# simple imputer for zeroes features - strategy 'median'

#df = df.replace(0, df.median())
# checking for 'null' data in dataframe

#df.isnull().sum().sort_values(ascending = False).head(20)
# removing of features with nan-values more than 50% 

#df = df.loc[:, (df.isnull().sum(axis=0) <= len(df.index)*0.5)]
#s = (df.dtypes == 'object')

#object_cols = list(s[s].index)

#object_cols
#for col in columns:

    #df[col].fillna(df[col].mode().values[0], inplace = True)
# replace NaN to MODE

#from statistics import mode

#df = df.fillna(mode)
'''

s = (df.dtypes == 'object')

object_cols = list(s[s].index)

object_cols

'''

# Apply label encoder

from sklearn.preprocessing import LabelEncoder

from category_encoders import TargetEncoder

from category_encoders.cat_boost import CatBoostEncoder

le = LabelEncoder()

#le = TargetEncoder(cols=object_cols)

#le = CatBoostEncoder(cols=object_cols)

for col in object_columns:

    object_columns[col] = le.fit_transform(object_columns[col].astype(str))
#from eli5.sklearn import PermutationImportance
df = pd.concat([object_columns, numeric_columns], axis=1)
df.head()
'''

import itertools

from sklearn import preprocessing, metrics

interactions = pd.DataFrame(index=df.index)

for col1, col2 in itertools.combinations(object_cols, 2):

    new_col_name = '_'.join([col1, col2])



    # Convert to strings and combine

    new_values = df[col1].map(str) + "_" + df[col2].map(str)



    encoder = preprocessing.LabelEncoder()

    interactions[new_col_name] = encoder.fit_transform(new_values)

df = df.join(interactions)

'''
df.corr(method='pearson', min_periods=1).tail(60)
#Correlation with output variable

cor = df.corr()

cor_target = (cor['SalePrice'])

#Selecting highly correlated features

relevant_features = cor_target[(cor_target<=-0.03) | (cor_target>=0.03) ]

relevant_features.sort_values(ascending = False).tail(60)

#correlation matrix

import seaborn as sns

import matplotlib.pyplot as plt  # Matlab-style plotting

corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix

k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
features = relevant_features.keys().tolist()

features
#df = pd.DataFrame(df, columns = features)

df = df[features]

df
df = df.reset_index()

#ignore 'error' in case of feature is not in frame

X = df.drop(['SalePrice'], axis=1, errors = 'ignore')



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=220)
from sklearn import metrics

from sklearn.model_selection import cross_val_score



def cross_val(model):

    pred = cross_val_score(model, X, y, cv=10)

    return pred.mean()



def print_evaluate(true, predicted):  

    mae = metrics.mean_absolute_error(true, predicted)

    mse = metrics.mean_squared_error(true, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))

    r2_square = metrics.r2_score(true, predicted)

    print('MAE:', mae)

    print('MSE:', mse)

    print('RMSE:', rmse)

    print('R2 Square', r2_square)

    

def evaluate(true, predicted):

    mae = metrics.mean_absolute_error(true, predicted)

    mse = metrics.mean_squared_error(true, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))

    r2_square = metrics.r2_score(true, predicted)

    return mae, mse, rmse, r2_square
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression(normalize=True)

lin_reg.fit(X_train,y_train)
# print the intercept

print(lin_reg.intercept_)
coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])

coeff_df
pred = lin_reg.predict(X_test)
plt.scatter(y_test, pred)
sns.distplot((y_test - pred), bins=50);
print_evaluate(y_test, lin_reg.predict(X_test))
results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, pred) , cross_val(LinearRegression())]], 

                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

results_df
from sklearn.linear_model import RANSACRegressor



model = RANSACRegressor()

model.fit(X_train, y_train)



pred = model.predict(X_test)

print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["Robust Regression", *evaluate(y_test, pred) , cross_val(RANSACRegressor())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.linear_model import Ridge



model = Ridge()

model.fit(X_train, y_train)

pred = model.predict(X_test)



print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, pred) , cross_val(Ridge())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.linear_model import Lasso



model = Lasso()

model.fit(X_train, y_train)

pred = model.predict(X_test)



print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, pred) , cross_val(Lasso())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.linear_model import ElasticNet



model = ElasticNet()

model.fit(X_train, y_train)

pred = model.predict(X_test)



print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_test, pred) , cross_val(ElasticNet())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.tree import DecisionTreeRegressor



model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

model.fit(X_train, y_train)

pred = model.predict(X_test)



print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["Decision Tree", *evaluate(y_test, pred) , cross_val(DecisionTreeRegressor())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=50, random_state=2, max_leaf_nodes=230)

model.fit(X_train, y_train)

pred = model.predict(X_test)



print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["Random Forest", *evaluate(y_test, pred) , cross_val(RandomForestRegressor())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from xgboost import XGBRegressor



model = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)],

             verbose=False)

pred = model.predict(X_test)



print_evaluate(y_test, pred)



''''''
results_df_2 = pd.DataFrame(data=[["XGBR", *evaluate(y_test, pred) , cross_val(XGBRegressor())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df


import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state=1).fit(X_train, y_train)

eli5.show_weights(perm, feature_names = X_train.columns.tolist())
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =42)

model.fit(X_train, y_train)

pred = model.predict(X_test)



print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["GradientBoostingRegressor", *evaluate(y_test, pred) , cross_val(GradientBoostingRegressor())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state=1).fit(X_train, y_train)

eli5.show_weights(perm, feature_names = X_train.columns.tolist())
model = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       #min_data_in_leaf=2,

                                       #min_sum_hessian_in_leaf=11

                                       )

model.fit(X_train, y_train)

pred = model.predict(X_test)



print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["LGBMRegressor", *evaluate(y_test, pred) , cross_val(LGBMRegressor())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state=1).fit(X_train, y_train)

eli5.show_weights(perm, feature_names = X_train.columns.tolist())