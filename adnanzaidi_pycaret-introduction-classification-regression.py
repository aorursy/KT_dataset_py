!pip install pycaret
import numpy as np 
import pandas as pd 
train = pd.read_csv('../input/titanic/train.csv')
test  = pd.read_csv('../input/titanic/test.csv')
sub   = pd.read_csv('../input/titanic/gender_submission.csv')
from pycaret.classification import *
train.head()
train.info()
clf1 = setup(data = train, 
             target = 'Survived',
             numeric_imputation = 'mean',
             categorical_features = ['Sex','Embarked'], 
             ignore_features = ['Name','Ticket','Cabin'],
             silent = True)

#quite intuitive isn't it ?
compare_models()
lgbm  = create_model('lightgbm')      
tuned_lightgbm = tune_model('lightgbm')
plot_model(estimator = tuned_lightgbm, plot = 'learning')
plot_model(estimator = tuned_lightgbm, plot = 'auc')
plot_model(estimator = tuned_lightgbm, plot = 'confusion_matrix')
plot_model(estimator = tuned_lightgbm, plot = 'feature')
evaluate_model(tuned_lightgbm)
interpret_model(tuned_lightgbm)
predict_model(tuned_lightgbm, data=test)
predictions = predict_model(tuned_lightgbm, data=test)
predictions.head()
sub['Survived'] = round(predictions['Score']).astype(int)
sub.to_csv('submission.csv',index=False)
sub.head()
logr  = create_model('lr');      
xgb   = create_model('xgboost');            

#blending 3 models
blend = blend_models(estimator_list=[tuned_lightgbm,logr,xgb])
from pycaret.regression import *
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample= pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.info()
reg = setup(data = train, 
             target = 'SalePrice',
             numeric_imputation = 'mean',
             categorical_features = ['MSZoning','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType',
                                     'Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',   
                                     'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',    
                                     'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',   
                                     'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',   
                                     'Electrical','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
                                     'SaleCondition']  , 
             ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],
             normalize = True,
             silent = True)
compare_models()
cb = create_model('catboost')
tuned_cb = tune_model('catboost')
interpret_model(tuned_cb)
predictions = predict_model(tuned_cb, data = test)
sample['SalePrice'] = predictions['Label']
sample.to_csv('submission_house_price.csv',index=False)
sample.head()