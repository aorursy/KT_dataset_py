import numpy as np

import pandas as pd 

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col=0)

test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col=0)



X = pd.concat([train.drop("SalePrice", axis=1),test], axis=0)

y = train[['SalePrice']]
def df_preprocessing(df):

    

    ## Dropping Features that are not useful for model prediction

    df.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'], axis=1, inplace=True)    #drop highly correlated feature

    df.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)         #drop top 3 columns with most number of missing values

    df.drop(['MoSold','YrSold'], axis=1, inplace=True)          #remove columns with no relationship with SalePrice

    

    df_col = df.columns     #remove columns with >96% same values

    overfit_col = []

    for i in df_col:

        counts = df[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(X) * 100 > 96:

            overfit_col.append(i)



    overfit_col = list(overfit_col)

    df = df.drop(overfit_col, axis=1)



    

    ## Removing outliers

    global train

    train = train.drop(train[train['LotFrontage'] > 200].index)

    train = train.drop(train[train['LotArea'] > 100000].index)

    train = train.drop(train[train['BsmtFinSF1'] > 4000].index)

    train = train.drop(train[train['TotalBsmtSF'] > 5000].index)

    train = train.drop(train[train['GrLivArea'] > 4000].index)

    

    

    ## Impute missing values

    ordd = ['GarageType','GarageFinish','BsmtFinType2','BsmtExposure','BsmtFinType1', 

       'GarageCond','GarageQual','BsmtCond','BsmtQual','FireplaceQu','Fence',"KitchenQual",

       "HeatingQC",'ExterQual','ExterCond']

    df[ordd] = df[ordd].fillna("NA")         #Ordinal columns replace missing values with NA

    

    cat = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "Functional"]

    df[cat] = df.groupby("Neighborhood")[cat].transform(lambda x: x.fillna(x.mode()[0]))      #Nominal columns replace missing value with most frequent occurrence aka mode

    

    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))      #Replace with mean after grouping by Neighborhood

    df['GarageArea'] = df.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean())) 

    df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



    cont = ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea"]

    df[cont] = X[cont] = X[cont].fillna(X[cont].mean())       #Replace missing values with respective mean values for continuous features

    

    

    ## Mapping Ordinal Features

    ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}

    fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}

    expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}

    fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}

    

    ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']

    for col in ord_col:

        df[col] = df[col].map(ordinal_map)



    fin_col = ['BsmtFinType1','BsmtFinType2']

    for col in fin_col:

        df[col] = df[col].map(fintype_map)



    df['BsmtExposure'] = df['BsmtExposure'].map(expose_map)

    df['Fence'] = df['Fence'].map(fence_map)

    

    ## Change data type

    df['MSSubClass'] = df['MSSubClass'].apply(str)

    

    return df





## Feature Engineering

def feat_engineer(df):

    

    ## Add new features based on merging relevant existing features

    X['TotalLot'] = X['LotFrontage'] + X['LotArea']

    X['TotalBsmtFin'] = X['BsmtFinSF1'] + X['BsmtFinSF2']

    X['TotalSF'] = X['TotalBsmtSF'] + X['2ndFlrSF']

    X['TotalBath'] = X['FullBath'] + X['HalfBath']

    X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['ScreenPorch']

    

    

    ## Generate Binary columns indicating 0/1 the presence of such features

    colum = ['MasVnrArea','TotalBsmtFin','TotalBsmtSF','2ndFlrSF','WoodDeckSF','TotalPorch']

    for col in colum:

        col_name = col+'_bin'

        df[col_name] = df[col].apply(lambda x: 1 if x > 0 else 0)

    

    

    ## Convert categorical to numerical through One-hot encoding

    df = pd.get_dummies(df)

    return df





X = df_preprocessing(X)

X = feat_engineer(X)
## Scaling

cols = X.select_dtypes(np.number).columns

X[cols] = RobustScaler().fit_transform(X[cols])



## Log transformation of SalePrice

y["SalePrice"] = np.log(y['SalePrice'])



## Return train and test index

x = X.loc[train.index]

y = y.loc[train.index]

test = X.loc[test.index]



#Split train into train/validation set for training

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2020)



print(X_train.shape)
from sklearn.ensemble import GradientBoostingRegressor



gbr = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, criterion='friedman_mse',

                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 

                                min_impurity_split=None, random_state=2020, max_features='sqrt', 

                                alpha=0.9, tol=0.0001)   #default from documentation
#Training

gbr.fit(X_train, y_train.values.ravel())



#Inference

y_pred = gbr.predict(X_val)



#Evaluate

np.sqrt(mean_squared_error(y_val,y_pred))
import xgboost as xgboost



xgb = xgboost.XGBRegressor(n_estimators=500,

                           max_depth=3,

                           learning_rate=0.1,

                           min_child_weight=2,

                           grow_policy='lossguide',

                           random_state=2020)
#Training

xgb.fit(X_train, y_train.values.ravel())



#Inference

y_pred = xgb.predict(X_val)



#Evaluate

np.sqrt(mean_squared_error(y_val,y_pred))
import lightgbm as lightgbm



lgb = lightgbm.LGBMRegressor(boosting_type='gbdt',

                             max_depth=3,

                             learning_rate=0.1,

                             n_estimators=500,

                             min_child_samples=2,

                             reg_alpha=0.0,

                             reg_lambda=0.1,

                             random_state=2020

                             )
#Training

lgb.fit(X_train, y_train.values.ravel())



#Inference

y_pred = lgb.predict(X_val)



#Evaluate

np.sqrt(mean_squared_error(y_val,y_pred))
from catboost import CatBoostRegressor



cb = CatBoostRegressor(max_depth=3,

                       learning_rate=0.1,

                       n_estimators=500,

                       loss_function='RMSE',

                       boosting_type='Ordered',

                       min_child_samples=2,

                       l2_leaf_reg=0.1,

                       random_state=2020,

                       logging_level='Silent')
#Training

cb.fit(X_train, y_train.values.ravel())



#Inference

y_pred = cb.predict(X_val)



#Evaluate

np.sqrt(mean_squared_error(y_val,y_pred))
from sklearn.model_selection import GridSearchCV



learning_rate = [0.0001, 0.001, 0.01, 0.1]

n_estimators = [50, 100, 250, 500]

max_depth = [3,5,10]



param_grid = dict(learning_rate = learning_rate, #Dictionary with parameters names (str) as keys and lists of parameter settings to try as values

             n_estimators = n_estimators,

             max_depth = max_depth)



grid = GridSearchCV(estimator=gbr,

                    param_grid=param_grid,

                    scoring="neg_root_mean_squared_error",

                    verbose=1,

                    n_jobs=-1)



grid_gbr = grid.fit(X_train,y_train)

print('Best Score: ', grid_gbr.best_score_)

print('Best Params: ', grid_gbr.best_params_)
from sklearn.model_selection import RandomizedSearchCV



param_distributions = {'learning_rate' : [0.0001, 0.001, 0.01, 0.1],

                       'n_estimators' : [50, 100, 250, 500, 700, 900],

                       'max_depth' : [i for i in range(10)],

                       'min_samples_split': [2, 5, 10, 20, 40],

                       'max_features' : ["auto", "sqrt", "log2"]}



rdm_grid = RandomizedSearchCV(estimator=gbr,

                             param_distributions = param_distributions,

                             n_iter = 100,

                             n_jobs = -1,

                             scoring="neg_root_mean_squared_error")



rdm_gbr = rdm_grid.fit(X_train, y_train)

print('Best Score: ', rdm_gbr.best_score_)

print('Best Params: ', rdm_gbr.best_params_)