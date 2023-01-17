!pip install pycaret

!pip install pandas_profiling
import pandas as pd 

import pandas_profiling as pp
train_house=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_house=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
pp.ProfileReport(train_house)
from pycaret.regression import *
regression_setup =setup(data = train_house, 

             target = 'SalePrice',

             numeric_imputation = 'mean', #fill missing value with mean for numeric features

             categorical_features = ['MSZoning','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType',

                                     'Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',   

                                     'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',    

                                     'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',   

                                     'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',   

                                     'Electrical','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',

                                     'SaleCondition'], #categorical features from pandas profiling report

             ignore_features = ['Id'],

             train_size=0.8,

             normalize=True,

             normalize_method='minmax',

             handle_unknown_categorical=True,

             unknown_categorical_method='most_frequent',  #fill missing value with most frequent value for categorical features

             remove_outliers=True, #it automatically applies PCA for removing outliers,

             outliers_threshold=0.05, 

             silent=True,

             profile=True #a data profile for Exploratory Data Analysis will be displayed in an interactive HTML report. It also generates pandas profiling report

     )
compare_models(blacklist = ['ransac', 'tr', 'rf', 'et', 'ada', 'gbr'])
lgbm = create_model(

    estimator='lightgbm',

    fold=5

)
evaluate_model(lgbm)
lgbm_holdout_pred = predict_model(lgbm)
house_prediction =  predict_model(lgbm, data=test_house)

house_prediction.head()
house_prediction.rename(columns={'Label':'SalePrice'}, inplace=True)

house_prediction[['Id','SalePrice']].to_csv('submission_house.csv', index=False)