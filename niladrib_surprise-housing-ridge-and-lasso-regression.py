import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math

import statsmodels.api as sm

import datetime



%matplotlib inline



import seaborn as sns



import sklearn

import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso, Ridge

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



from sklearn.feature_selection import RFE



pd.options.display.max_colwidth = 200

pd.options.display.max_rows=100

pd.options.display.max_columns=500



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/surprise-housing-dataset/train.csv')
data.head()
data.shape
data.info()
data.describe().T
# checking null values

NA_col = data.isnull().sum()

# find out columns which have nulls

NA_col = NA_col[NA_col > 0]

# % of columns missing

print(round(100*NA_col[NA_col > 0]/len(data),2))
# Dropping columns which have high values missing + I

data.drop(['Id','LotFrontage','Alley','FireplaceQu','PoolQC', 'Fence', 'MiscFeature', 'MoSold','Street', 'Utilities'], axis=1, inplace = True)
data.shape
# checking null values

NA_col = data.isnull().sum()

# find out columns which have nulls

NA_col = NA_col[NA_col > 0]

# % of columns missing

print(round(100*NA_col[NA_col > 0]/len(data),2))
# Converting years to age

data['YearBuilt_Age'] = data['YearBuilt'].max() - data['YearBuilt']

data['YearRemodAdd_Age'] = data['YearRemodAdd'].max() - data['YearRemodAdd']

data['YrSold_Age'] = data['YrSold'].max() - data['YrSold']

data['GarageYrBlt_Age'] = data['GarageYrBlt'].max() - data['GarageYrBlt']



# Dropping columns

data.drop(['YearBuilt','YearRemodAdd','YrSold','GarageYrBlt'], axis=1, inplace = True)
data[['YearBuilt_Age','YearRemodAdd_Age','YrSold_Age','GarageYrBlt_Age']].head(10)
def treat_Missing_Values(df):    

    

    # checking null values

    NA_col = df.isnull().sum()

    # find out columns which have nulls

    NA_col = NA_col[NA_col > 0]



    for col in NA_col.index:

        if df[col].dtype.name == 'object':

            # impute mode

            df[col].fillna(data[col].mode()[0], inplace=True)

            

        elif df[col].dtype.name == 'float64' or df[col].dtype.name == 'int64' or df[col].dtype.name == 'int32':

            # impute median

            df[col] = df[col].fillna((df[col].median()))

            

        else:

            print('Unable to detect the datatype for col - ', col)

            

    return df
data = treat_Missing_Values(data)
# checking null values

round(data.isnull().sum()/len(data.index),2)[round(data.isnull().sum()/ len(data.index),2).values>0.00]
f, ax = plt.subplots(figsize=(30, 30))

sns.heatmap(data.corr(), 

            xticklabels=data.corr().columns.values,

            yticklabels=data.corr().columns.values,annot= True)



bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)



plt.show()
corr_val = data[list(data.dtypes[data.dtypes != 'object'].index)].corr()
corr_coef = corr_val[corr_val['SalePrice'] > 0.5]['SalePrice'].sort_values(ascending=False)

print(corr_coef[1:])

corr_coef_cols = [idx for idx in corr_coef.index]
# Draw Pair plot for the correlated features

sns.pairplot(data, x_vars=corr_coef_cols[1:], y_vars=[corr_coef_cols[0]], kind="reg" )
# Let us check the SalePrice as well

f, axes = plt.subplots(1, 2, figsize=(15,6))

sns.boxplot(data['SalePrice'],  orient='v' , ax=axes[0])

sns.distplot(data['SalePrice'], ax=axes[1])

plt.show()
def label_encoding(colNames):

    for colName in colNames:

        unique_vals = data[colName].unique()

        map_vals = {}

        for idx, val in enumerate(unique_vals):

            map_vals[val] = idx

        data[colName] = data[colName].map(map_vals)
cat_col_list = ['LandSlope', 'ExterQual', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 

                'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 

                'GarageFinish', 'GarageQual', 'GarageCond', 'ExterCond', 'LotShape']

label_encoding(cat_col_list)
data[cat_col_list].head()
dummy_col_names = ['MSZoning','LandContour','LotConfig','Neighborhood','Condition1','Condition2','BldgType',

             'HouseStyle','RoofStyle','RoofMatl','Exterior1st', 'Exterior2nd','MasVnrType','Foundation',

             'Heating','Electrical','Functional','GarageType','PavedDrive','SaleType','SaleCondition']

dummies = pd.get_dummies(data[dummy_col_names], drop_first = True)

data = pd.concat([data, dummies], axis = 1)

data.drop(dummy_col_names, axis = 1, inplace = True)
data.head()
data.shape
# Transform SalePrice

data['SalePrice'] = np.log1p(data['SalePrice'])
# Create train and test data

df_train, df_test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=100)
# Scale data

scaler_col = ['MSSubClass','LotArea','OverallQual','OverallCond', 'MasVnrArea','BsmtFinSF1',

              'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF', 'LowQualFinSF',

              'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',

              'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF',

              'OpenPorchSF','EnclosedPorch','3SsnPorch', 'ScreenPorch','PoolArea','MiscVal','SalePrice']



scaler = StandardScaler()

df_train[scaler_col] = scaler.fit_transform(df_train[scaler_col])

df_test[scaler_col] = scaler.transform(df_test[scaler_col])
df_train.head()
# Let us check if the target variable is normal in both train and test dataset

plt.figure(figsize=(16,6))

plt.subplot(121)

sns.distplot(df_train.SalePrice)

plt.subplot(122)

sns.distplot(df_test.SalePrice)
# Create X and y

y_train = df_train.pop('SalePrice')

X_train = df_train



y_test = df_test.pop('SalePrice')

X_test = df_test
# RFE

lm  = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm,50)

rfe.fit(X_train,y_train)
rfe_scores = pd.DataFrame(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))
col = X_train.columns[rfe.support_]
col
# Modify the X_train and X_test

X_train = X_train[col]

X_test = X_test[col]
X_train.shape
X_test.shape
#Lasso

lm = Lasso(alpha=0.001)

lm.fit(X_train,y_train)



# train score

y_train_pred = lm.predict(X_train)

print(metrics.r2_score(y_true=y_train,y_pred=y_train_pred))



# test score

y_test_pred  = lm.predict(X_test)

print(metrics.r2_score(y_true=y_test,y_pred=y_test_pred))
model_parameter = list(lm.coef_)

model_parameter.insert(0,lm.intercept_)

model_parameter = [round(x,3) for x in model_parameter]

col = X_train.columns

col = col.insert(0,'Constant')

list(zip(col,model_parameter))
# Gridsearch Operation on Training data set

# Objective - Find optimal value of alpha



folds = KFold(n_splits=10,shuffle=True,random_state=100)



hyper_param = {'alpha':[0.001, 0.01, 0.1,1.0, 5.0, 10.0,20.0]}



model = Lasso()



model_cv = GridSearchCV(estimator = model,

                        param_grid=hyper_param,

                        scoring='r2',

                        cv=folds,

                        verbose=1,

                        return_train_score=True

                       )



model_cv.fit(X_train,y_train)
cv_result_train_lasso = pd.DataFrame(model_cv.cv_results_)

cv_result_train_lasso['param_alpha'] = cv_result_train_lasso['param_alpha'].astype('float32')

cv_result_train_lasso.head()
def r2_score(cv_result, is_log=False):

    plt.figure(figsize=(12,6))

    plt.plot(cv_result['param_alpha'], cv_result['mean_train_score'])

    plt.plot(cv_result['param_alpha'], cv_result['mean_test_score'])

    if is_log == True:

        plt.xscale('log')

    plt.ylabel('R2 Score')

    plt.xlabel('Alpha')

    plt.show()
r2_score(cv_result_train_lasso, True)
print('For Lasso, the Best Alpha value = ', model_cv.best_params_['alpha'])
# Now that we have optimal value of alpha = 0.001, we can use this alpha to run the model again

#Lasso

lm = Lasso(alpha=0.001)

lm.fit(X_train,y_train)



# train score

y_train_pred = lm.predict(X_train)

print(metrics.r2_score(y_true=y_train,y_pred=y_train_pred))



# test score

y_test_pred  = lm.predict(X_test)

print(metrics.r2_score(y_true=y_test,y_pred=y_test_pred))
#Ridge

ridge = Ridge(alpha=0.001)

ridge.fit(X_train,y_train)



# train score

y_train_pred = ridge.predict(X_train)

print(metrics.r2_score(y_train, y_train_pred))



# test score

y_test_pred = ridge.predict(X_test)

print(metrics.r2_score(y_test, y_test_pred))
# Gridsearch Operation on Training data set

# Objective - Find optimal value of alpha



folds = KFold(n_splits=10,shuffle=True,random_state=100)



hyper_param = {'alpha':[0.001, 0.01, 0.1,1.0, 5.0, 10.0,20.0]}



model = Ridge()



model_cv = GridSearchCV(estimator = model,

                        param_grid=hyper_param,

                        scoring='r2',

                        cv=folds,

                        verbose=1,

                        return_train_score=True

                       )



model_cv.fit(X_train,y_train)
cv_result_train_ridge = pd.DataFrame(model_cv.cv_results_)

cv_result_train_ridge['param_alpha'] = cv_result_train_ridge['param_alpha'].astype('float32')

cv_result_train_ridge.head()
# plot r2_score using the defined function for ridge

r2_score(cv_result_train_ridge, True)
print('For Ridge, the Best Alpha value = ', model_cv.best_params_['alpha'])
# Now that we have optimal value of alpha = 0.001, we can use this alpha to run the model again

#Ridge

ridge = Ridge(alpha=10)

ridge.fit(X_train,y_train)



# train score

y_train_pred = ridge.predict(X_train)

print(metrics.r2_score(y_train, y_train_pred))



# test score

y_test_pred = ridge.predict(X_test)

print(metrics.r2_score(y_test, y_test_pred))
# ridge coefficients

model_parameter = list(ridge.coef_)

model_parameter.insert(0,ridge.intercept_)

model_parameter = [round(x,3) for x in model_parameter]

col = X_train.columns

col = col.insert(0,'Constant')

list(zip(col,model_parameter))
def run_multiple_alphas(model,alphas):

    

    feature_ridge_df = pd.DataFrame(columns=['feature'], data=X_train.columns)

    feature_lasso_df = pd.DataFrame(columns=['feature'], data=X_train.columns)

    

    for alpha in alphas:

        if model == 'ridge':

            ridge = Ridge(alpha=alpha)

            ridge.fit(X_train, y_train)

            # Creating feature/coefficient map for future use

            feature_ridge_df['Alpha: '+str(alpha)] = ridge.coef_

        elif model == 'lasso':

            lasso = Lasso(alpha=alpha)

            lasso.fit(X_train, y_train)

            # Creating feature/coefficient map for future use

            feature_lasso_df['Alpha: '+str(alpha)] = lasso.coef_

    

    if model == 'ridge':

        return feature_ridge_df

    else:

        return feature_lasso_df
lasso_df = run_multiple_alphas('lasso',[0.001,0.002,0.01,0.02,0.05,5])
lasso_df.head()
print(lasso_df[lasso_df['Alpha: 0.001'] == 0][['feature', 'Alpha: 0.001']].shape)

print(lasso_df[lasso_df['Alpha: 0.002'] == 0][['feature', 'Alpha: 0.002']].shape)

print(lasso_df[lasso_df['Alpha: 0.01'] == 0][['feature', 'Alpha: 0.01']].shape)

print(lasso_df[lasso_df['Alpha: 0.02'] == 0][['feature', 'Alpha: 0.02']].shape)

print(lasso_df[lasso_df['Alpha: 0.05'] == 0][['feature', 'Alpha: 0.05']].shape)

print(lasso_df[lasso_df['Alpha: 5'] == 0][['feature', 'Alpha: 5']].shape)
# We know alpha = 0.001 is optimal value

lasso_df = lasso_df[['feature','Alpha: 0.001', 'Alpha: 0.002']]

lasso_df = lasso_df.reindex(lasso_df['Alpha: 0.002'].abs().sort_values(ascending=False).index)

lasso_df['predictors'] = lasso_df['feature'].apply(lambda x:x.split('_')[0])
lasso_df.head(10)
x = lasso_df[['feature','Alpha: 0.002','predictors']]
ridge_df = run_multiple_alphas('ridge',[10,20])
ridge_df.head()
# We know alpha = 10 is optimal value

ridge_df = ridge_df[['feature','Alpha: 10','Alpha: 20']]

ridge_df = ridge_df.reindex(ridge_df['Alpha: 20'].abs().sort_values(ascending=False).index)

ridge_df['predictors'] = ridge_df['feature'].apply(lambda x:x.split('_')[0])
ridge_df.head(10)