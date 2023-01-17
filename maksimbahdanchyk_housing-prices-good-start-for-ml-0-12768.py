import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv').drop('Id',axis=1)





test  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_id = test['Id']

test  = test.drop('Id',axis=1)
train = train.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

test  = test.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



obj_col = train.select_dtypes('object').columns

num_col = train.select_dtypes(exclude='object').columns[:-1]
ohe_col = ['MSZoning','Alley','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',

           'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',

           'MasVnrType','Foundation','Heating','Electrical','Functional','GarageType','GarageFinish','PavedDrive',

           'Fence','MiscFeature','SaleType','SaleCondition']



ord_col = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

           'HeatingQC','CentralAir','KitchenQual','FireplaceQu','GarageQual','GarageCond']
for col in obj_col:

    train[col] = train[col].fillna('No info')

    test[col] = test[col].fillna(train[col].value_counts().index[0])
for col in num_col:

    train[col] = train[col].fillna(train[col].mean())

    test[col] = test[col].fillna((train[col].mean()+test[col].mean())/2)
print(train.isna().sum()[train.isna().sum() != 0])

print(test.isna().sum()[test.isna().sum() != 0])
train.SalePrice.hist(bins = 100)
train['SalePrice']=train.SalePrice.apply(lambda x: np.log1p(x))
train.SalePrice.hist(bins = 100)
#------------------------------------------------------------------------------

# accept a dataframe, remove outliers, return cleaned data in a new dataframe

# see http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm

#------------------------------------------------------------------------------

def remove_outliers(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

    return df_out



train = remove_outliers(train,'SalePrice')

train.SalePrice.hist(bins = 100)
from sklearn.model_selection import train_test_split, cross_val_score,KFold, GridSearchCV

from sklearn.metrics         import mean_squared_log_error, mean_squared_error



from sklearn.pipeline        import make_pipeline

from sklearn.compose         import make_column_transformer



from sklearn.preprocessing   import OrdinalEncoder, OneHotEncoder

from sklearn.preprocessing   import MinMaxScaler, StandardScaler, RobustScaler



from sklearn.linear_model    import LinearRegression,Lasso,Ridge,LogisticRegression

from sklearn.ensemble        import RandomForestRegressor,GradientBoostingRegressor

from sklearn.ensemble        import StackingRegressor

from sklearn.svm             import SVR

import xgboost                   as xgb
trans = make_column_transformer(

                                (StandardScaler(),list(num_col)),

                                (OrdinalEncoder(),list(ord_col)),

                                (OneHotEncoder(), list(ohe_col)),

                                remainder = 'passthrough'

                                )



X = train.drop(['SalePrice'],axis = 1)

y = train['SalePrice']



X = trans.fit_transform(X)



X_train,X_val,y_train,y_val = train_test_split(X,y,

                                               test_size = 0.001,

                                               random_state = 42,

                                               shuffle = True)


model_gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4,

                                max_features='sqrt', min_samples_leaf=15, min_samples_split=10,

                                loss='huber', random_state =42)         



model_xgb = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3500,

                                     max_depth=10, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
folds = KFold(n_splits = 10, shuffle = True, random_state=42)

#score_las = np.sqrt(-cross_val_score(Lasso(),X_train,y_train,scoring = 'neg_mean_squared_error', cv=folds))

#score_rid = np.sqrt(-cross_val_score(Ridge(),X_train,y_train,scoring = 'neg_mean_squared_error', cv=folds))

#score_svr = np.sqrt(-cross_val_score(SVR(),X_train,y_train,scoring = 'neg_mean_squared_error', cv=folds))



#score_gbr = np.sqrt(-cross_val_score(model_gbr,X_train,y_train,scoring = 'neg_mean_squared_error', cv=folds))

#score_xgb = np.sqrt(-cross_val_score(model_xgb,X_train,y_train,scoring = 'neg_mean_squared_error', cv=folds))



#print('(LAS) Train CV mean score: ',np.mean(score_las),' STD: ',np.std(score_las))

#print('(RID) Train CV mean score: ',np.mean(score_rid),' STD: ',np.std(score_rid))

#print('(GBR) Train CV mean score: ',np.mean(score_gbr),' STD: ',np.std(score_gbr))

#print('(XGB) Train CV mean score: ',np.mean(score_xgb),' STD: ',np.std(score_xgb))
param_las = {'alpha':[0.0006,0.0007,0.0008,0.0009,0.001]}

las_cv = GridSearchCV(Lasso(random_state=42),param_las,scoring = 'neg_mean_squared_error',cv = folds)

las_cv.fit(X_train,y_train)

print('Best score: ',np.sqrt(-las_cv.best_score_),' With parameters: ',las_cv.best_params_)

model_las = las_cv.best_estimator_
param_rid = {'alpha':range(20,30)}

rid_cv = GridSearchCV(Ridge(random_state=42),param_rid,scoring = 'neg_mean_squared_error',cv = folds,)

rid_cv.fit(X_train,y_train)

print('Best score: ',np.sqrt(-rid_cv.best_score_),' With parameters: ',rid_cv.best_params_)

model_rid = rid_cv.best_estimator_
param_svr = {'C':[1,2,3,5,7,9],

             'gamma':[0.001,0.003,0.005,0.007]}

svr_cv = GridSearchCV(SVR(),param_svr,scoring = 'neg_mean_squared_error',cv = folds,)

svr_cv.fit(X_train,y_train)

print('Best score: ',np.sqrt(-svr_cv.best_score_),' With parameters: ',svr_cv.best_params_)

model_svr = svr_cv.best_estimator_
model_las.fit(X,y)

model_rid.fit(X,y)

model_svr.fit(X,y)

model_xgb.fit(X,y)

model_gbr.fit(X,y)





print('(LAS) RMSE on VAL: ', np.sqrt(mean_squared_error(y_val,model_las.predict(X_val))))

print('(RID) RMSE on VAL: ', np.sqrt(mean_squared_error(y_val,model_rid.predict(X_val))))

print('(SVR) RMSE on VAL: ', np.sqrt(mean_squared_error(y_val,model_svr.predict(X_val))))

print('(XGB) RMSE on VAL: ', np.sqrt(mean_squared_error(y_val,model_xgb.predict(X_val))))

print('(GBR) RMSE on VAL: ', np.sqrt(mean_squared_error(y_val,model_gbr.predict(X_val))))
estimators = [('LAS', model_las),

              ('RID', model_rid),

              ('SVR', model_svr),

              ('XGB',model_xgb),

              ('GBR',model_gbr)]



stack = StackingRegressor(estimators=estimators,final_estimator = model_xgb)

stack.fit(X_train,y_train)
def blend_models_predict(X):

    return ((0.05 * model_las.predict(X)) + \

            (0.05 * model_rid.predict(X)) + \

            (0.1 * model_svr.predict(X)) + \

            (0.2 * model_xgb.predict(X)) + \

            (0.2 * model_gbr.predict(X)) + \

            (0.4 * stack.predict(X)))

print('Blend model RMSE on VAL: ', np.sqrt(mean_squared_error(y,blend_models_predict(X))))
x = trans.transform(test)

pred = blend_models_predict(x)
submis = pd.DataFrame({'Id':test_id,'SalePrice':pred})

submis['SalePrice'] = np.expm1(submis['SalePrice'])

submis.to_csv('/kaggle/working/submission1.csv',index=False)
submis