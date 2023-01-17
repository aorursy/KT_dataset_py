import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pickle
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print("Train Shape: ",train.shape)

print("Test Shape: ",test.shape)
print(train.info())
print(test.info())
print(train.isnull().sum())
sns.heatmap(train.isnull())
print(test.isnull().sum())
sns.heatmap(test.isnull())
cat_col_train = ['FireplaceQu','GarageType','GarageFinish','MasVnrType','BsmtQual',

           'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',

          'GarageQual','GarageCond']



ncat_col_train = ['LotFrontage','GarageYrBlt','MasVnrArea']
for i in cat_col_train:

    train[i] = train[i].fillna(train[i].mode()[0])

    

for j in ncat_col_train:

    train[j] = train[j].fillna(train[j].mean())
cat_col_test = ['FireplaceQu','GarageType','GarageFinish','MasVnrType','BsmtQual',

           'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',

          'GarageQual','GarageCond','MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType']



ncat_col_test = ['LotFrontage','GarageYrBlt','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath',

                'BsmtHalfBath','GarageCars','GarageArea']
for i in cat_col_test:

    test[i] = test[i].fillna(test[i].mode()[0])

    

for j in ncat_col_test:

    test[j] = test[j].fillna(test[j].mean())
to_drop = ['Id','Alley','PoolQC','Fence','MiscFeature']



for k in to_drop:

    train.drop([k], axis = 1, inplace = True)

    test.drop([k], axis = 1, inplace = True)
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())
print("Train Shape: ",train.shape)

print("Test Shape: ",test.shape)
final_df = pd.concat([train,test], axis = 0)
final_df.shape
all_cat_col = ['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',

              'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',

              'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',

              'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',

              'Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',

              'GarageCond','PavedDrive','SaleType','SaleCondition']
def cat_onehot_encoding(multicol):

    df_final = final_df

    i = 0

    for fields in multicol:

        print(fields)

        df1 = pd.get_dummies(final_df[fields],drop_first = True)

        

        final_df.drop([fields], axis = 1, inplace = True)

        if i==0:

            df_final = df1.copy()

        else:

            df_final = pd.concat([df_final,df1], axis=1)

        i = i+1

    

    df_final = pd.concat([final_df,df_final], axis = 1)

    

    return df_final
final_df = cat_onehot_encoding(all_cat_col)
final_df.shape
final_df = final_df.loc[:,~final_df.columns.duplicated()]

final_df.shape
df_train = final_df.iloc[:1460,:]

df_test = final_df.iloc[1460:,:]
df_test.drop(['SalePrice'], axis = 1, inplace = True)
print("Train Shape: ",df_train.shape)

print("Test Shape: ",df_test.shape)
x_train = df_train.drop(['SalePrice'], axis = 1)

y_train = df_train['SalePrice']
import xgboost



xgb_model = xgboost.XGBRegressor()

xgb_model.fit(x_train, y_train)
param = {

    'n_estimators': [100, 500, 900, 1100, 1500],

    'max_depth': [2,3,5,10,15],

    'learning_rate': [0.05, 0.1, 0.15, 0.2],

    'min_child_weight': [1,2,3,4],

    'booster': ['gbtree','gblinear'],

    'base_score': [0.25, 0.5, 0.75, 1]

}
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator=xgb_model,

                              param_distributions = param,

                              cv=5, n_iter=50,

                              scoring = 'neg_mean_absolute_error', n_jobs = 4,

                              verbose = 5,

                              return_train_score = True,

                              random_state = 42)

random_cv.fit(x_train, y_train)
random_cv.best_estimator_
xgb_model = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.1, max_delta_step=0, max_depth=2,

             min_child_weight=1, monotone_constraints='()',

             n_estimators=900, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)



xgb_model.fit(x_train, y_train)
f = "xgb_model.pkl"

pickle.dump(xgb_model,open(f,'wb'))
pred_xgb = xgb_model.predict(df_test)

print(pred_xgb.shape)
sub_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub_df['SalePrice'] = pred_xgb

sub_df.to_csv('sample_sub_xgb.csv', index = False)
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout
from keras import backend as k

def root_mean_squared_error(y_true, y_pred):

    return k.sqrt(k.mean(k.square(y_pred - y_true)))
nn_model = Sequential()



nn_model.add(Dense(50, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 176))

nn_model.add(Dense(25, kernel_initializer = 'he_uniform', activation = 'relu'))

nn_model.add(Dense(50, kernel_initializer = 'he_uniform', activation = 'relu'))

nn_model.add(Dense(1, kernel_initializer = 'he_uniform'))



nn_model.compile(loss = root_mean_squared_error, optimizer = 'Adamax')



nn_model.fit(x_train.values, y_train.values, validation_split = 0.25, batch_size = 10, epochs = 1000)
nn_model.save('nn_model.h5')
pred_nn = nn_model.predict(df_test)

print(pred_nn.shape)
sub_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub_df['SalePrice'] = pred_nn

sub_df.to_csv('sample_sub_nn.csv', index = False)