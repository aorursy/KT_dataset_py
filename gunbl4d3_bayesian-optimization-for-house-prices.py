import numpy as np 
import pandas as pd

import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import os
print(os.listdir("../input/previous-kernel-dataset"))
train = pd.read_csv("../input/previous-kernel-dataset/train_clean.csv")
test = pd.read_csv("../input/previous-kernel-dataset/test_clean.csv")

#Get ID and features
y_train = train['SalePrice']
train.drop(columns=['SalePrice', 'Id'], inplace=True)
test_id = test['Id']
test.drop(columns=['Id'], inplace=True)

print("Does Train feature equal test feature?: ", all(train.columns == test.columns))
#Convert target variables to logarithmic scale
y_train = np.log(y_train)
#Create LGBM dataset format. Need to convert string categorical variables to int.
def categorical_to_int(df):
    categorical = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                   'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                   'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 
                   'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                   'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
                   'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                   'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

    for col in categorical:
        df[col] = df[col].astype('category')
    
    df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
    
    return df

ntrain = train.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data = categorical_to_int(all_data)
train = all_data[:ntrain]
test = all_data[ntrain:]

dtrain = lgb.Dataset(train, label=y_train, free_raw_data=False)
dtrain.construct()
def evaluate_lgbm(max_depth,num_leaves,min_data_in_leaf,eta,feature_fraction):
    params = {
        'task': 'train',
        'objective': 'regression',
        'categorical_feature': ("name:MSZoning,Street,Alley,LotShape,LandContour,Utilities," 
            "LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,RoofStyle,"
            "RoofMatl,Exterior1st,Exterior2nd,MasVnrType,ExterQual,ExterCond,Foundation,BsmtQual,"
            "BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinType2,Heating,HeatingQC,CentralAir,"
            "Electrical,KitchenQual,Functional,FireplaceQu,GarageType,GarageFinish,GarageQual,"
            "GarageCond,PavedDrive,PoolQC,Fence,MiscFeature,SaleType,SaleCondition"),
        'max_depth': int(max_depth),
        'num_leaves': int(num_leaves),
        'min_data_in_leaf': int(min_data_in_leaf),
        'eta': max(eta,0),
        'feature_fraction': max(min(feature_fraction, 1), 0)
        }
    
    cv_results = lgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=5,
        metrics='rmse',
        early_stopping_rounds=10,
        stratified=False
        )
    
    #Return negative rmse, since bayesian optimization can only maximize
    return -1.0 * cv_results['rmse-mean'][-1] 
bayes_optim = BayesianOptimization(evaluate_lgbm, {'max_depth': (1,4),
                                                   'num_leaves': (2,10),
                                                   'min_data_in_leaf': (20,100),
                                                   'eta': (0.001,0.005),
                                                   'feature_fraction': (0.1,1)})
gp_params = {'alpha': 1e-5} #For convergence issues
bayes_optim.maximize(init_points=5, n_iter=25,**gp_params)

cv_params = bayes_optim.res['max']['max_params']
print(cv_params)
def submission_prediction(train,y_train,dtrain,test,cv_params):
    params = {
        'task': 'train',
        'objective': 'regression',
        'metric': 'rmse',
        'categorical_feature': ("name:MSZoning,Street,Alley,LotShape,LandContour,Utilities," 
            "LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,RoofStyle,"
            "RoofMatl,Exterior1st,Exterior2nd,MasVnrType,ExterQual,ExterCond,Foundation,BsmtQual,"
            "BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinType2,Heating,HeatingQC,CentralAir,"
            "Electrical,KitchenQual,Functional,FireplaceQu,GarageType,GarageFinish,GarageQual,"
            "GarageCond,PavedDrive,PoolQC,Fence,MiscFeature,SaleType,SaleCondition"),
        'max_depth': int(cv_params['max_depth']),
        'num_leaves': int(cv_params['num_leaves']),
        'min_data_in_leaf': int(cv_params['min_data_in_leaf']),
        'eta': max(cv_params['eta'],0),
        'feature_fraction': max(min(cv_params['feature_fraction'], 1), 0)
        }
    
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_preds = np.zeros(test.shape[0])
    oof_preds = np.zeros(train.shape[0])

    for train_idx, valid_idx in folds.split(train):
        mdl = lgb.train(
            params=params,
            train_set=dtrain.subset(train_idx),
            valid_sets=dtrain.subset(valid_idx),
            num_boost_round=1000, 
            early_stopping_rounds=10,
            verbose_eval=50
        )
        oof_preds[valid_idx] = mdl.predict(dtrain.data.iloc[valid_idx])
        fold_preds += mdl.predict(test) / folds.n_splits

        print("RMSE on validation set: %.5f" % 
              np.sqrt(mean_squared_error(y_train.iloc[valid_idx], oof_preds[valid_idx])))
        
    return fold_preds
y_pred = np.exp(submission_prediction(train,y_train,dtrain,test,cv_params))

submission = pd.DataFrame({
    "Id": test_id,
    "SalePrice": y_pred,
    })

submission.to_csv('house_prices.csv',index=False),
submission.head()
