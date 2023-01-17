import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.shape
plt.figure(figsize=(15,5))

plt.scatter(train.index,train.SalePrice.sort_values().reset_index(drop=True))

plt.title('Distribution Plot for Sales Prices')

plt.ylabel('Sales Price');
sns.heatmap(train.isnull(),yticklabels=False, cmap='plasma')
train.isnull().sum().sort_values(ascending=False)[0:19]
test.isnull().sum().sort_values(ascending=False)[0:33]
train.LotFrontage.head()
train.LotFrontage.isnull().sum()
train['LotFrontage'] = train['LotFrontage'].fillna(train.LotFrontage.mean())
test.LotFrontage.isnull().sum()
test['LotFrontage'] = test['LotFrontage'].fillna(test.LotFrontage.mean())
train.Alley.value_counts(dropna=False)
train.drop(columns=['Alley'], inplace=True)
test.Alley.value_counts(dropna=False)
test.drop(columns=['Alley'], inplace=True)
train.BsmtCond.value_counts(dropna=False)
train['BsmtCond'] = train['BsmtCond'].fillna(train.BsmtCond.mode()[0])
test['BsmtCond'] = test['BsmtCond'].fillna(test.BsmtCond.mode()[0])
list1 = ['BsmtQual', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'MasVnrType', 'MasVnrArea',

         'BsmtExposure','BsmtFinType2']



for item in list1:

    train[item] = train[item].fillna(train[item].mode()[0])

    test[item] = test[item].fillna(test[item].mode()[0])
list1 = ['GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature']



for item in list1:

    train.drop(columns=item, inplace=True)

    test.drop(columns=item, inplace=True)
train.isnull().sum().sort_values(ascending=False)
train.dropna(inplace=True)
train.drop(columns=['Id'], inplace=True)
train.shape
test.isnull().sum().sort_values(ascending=False)[0:17]
test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])
columns = ['BsmtFinType1', 'Utilities','BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 

           'Exterior1st', 'KitchenQual']

columns1 = ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'GarageArea']



for item in columns:

    test[item] = test[item].fillna(test[item].mode()[0])

for item in columns1:

    test[item] = test[item].fillna(test[item].mean())
test.drop(columns=['Id'], inplace=True)
test.shape
train.isnull().any().any()
test.isnull().any().any()
columns = ['MSZoning', 'Street',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical',

       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',

       'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
len(columns)
final_df = pd.concat([train, test], axis=0)
final_df.shape
def One_hot_encoding(columns):

    df_final=final_df

    i=0

    for fields in columns:

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:           

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([final_df,df_final],axis=1)

        

    return df_final
final_df = One_hot_encoding(columns)
final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
df_Train=final_df.iloc[:1422,:]

df_Test=final_df.iloc[1422:,:]
df_Test.drop(['SalePrice'],axis=1,inplace=True)
X_train=df_Train.drop(['SalePrice'],axis=1)

y_train=df_Train['SalePrice']
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X_train)



my_columns = X_train.columns

new_df = pd.DataFrame(X_std, columns=my_columns)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

df_pca = pca.fit_transform(new_df)
plt.figure(figsize =(8, 6))

plt.scatter(df_pca[:, 0], df_pca[:, 1], c = y_train, cmap ='plasma')

# labeling x and y axes

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component');
from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV



n_estimators = [100, 500, 900]

criterion = ['mse', 'mae']

depth = [3,5,10,15]

min_split=[2,3,4]

min_leaf=[2,3,4]

bootstrap = ['True', 'False']

verbose = [5]



hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':depth,

    'criterion':criterion,

    'bootstrap':bootstrap,

    'verbose':verbose,

    'min_samples_split':min_split,

    'min_samples_leaf':min_leaf

    }



random_cv = RandomizedSearchCV(estimator=regressor,

                               param_distributions=hyperparameter_grid,

                               cv=5, 

                               scoring = 'neg_mean_absolute_error',

                               n_jobs = 4, 

                               return_train_score = True,

                               random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor = RandomForestRegressor(bootstrap='True', ccp_alpha=0.0, criterion='mae',

                      max_depth=15, max_features='auto', max_leaf_nodes=None,

                      max_samples=None, min_impurity_decrease=0.0,

                      min_impurity_split=None, min_samples_leaf=2,

                      min_samples_split=4, min_weight_fraction_leaf=0.0,

                      n_estimators=100, n_jobs=None, oob_score=False,

                      random_state=None, verbose=5, warm_start=False)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(df_Test)
y_pred
pred=pd.DataFrame(y_pred)

samp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub = pd.concat([samp['Id'],pred], axis=1)

sub.columns=['Id','SalePrice']
sub
#sub.to_csv('My_sub.csv',index=False)
import xgboost
regressor=xgboost.XGBRegressor()
n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]

base_score=[0.25,0.5,0.75,1]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.1, max_delta_step=0, max_depth=2,

             min_child_weight=1, missing=None, monotone_constraints='()',

             n_estimators=900, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',

             validate_parameters=1, verbosity=None)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(df_Test)
y_pred
pred=pd.DataFrame(y_pred)

samp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub = pd.concat([samp['Id'],pred], axis=1)

sub.columns=['Id','SalePrice']
sub
#sub.to_csv('My_sub1.csv',index=False)