import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

globaldata = pd.concat([train,test])
pd.set_option('display.max_rows', 500)

((globaldata.isna().sum()/len(globaldata)*100).round(3))
globaldata.info()
!pip install pycaret

from pycaret.regression import *
reg= setup(data= train, target = 'SalePrice',train_size= 0.75,numeric_features=['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 

                               'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'PoolArea'],

                                ordinal_features= {'ExterQual': ['Fa', 'TA', 'Gd', 'Ex'],'ExterCond' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'BsmtQual' : ['Fa', 'TA', 'Gd', 'Ex'],'BsmtCond' : ['Po', 'Fa', 'TA', 'Gd'],'BsmtExposure' : ['No', 'Mn', 'Av', 'Gd'],

                                'HeatingQC' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],'KitchenQual' : ['Fa', 'TA', 'Gd', 'Ex'],'GarageQual' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'GarageCond' : ['Po', 'Fa', 'TA', 'Gd', 'Ex']},ignore_features = ['Alley','MasVnrType','MasVnrArea','FireplaceQu',"PoolQC","Fence","MiscFeature"],         

                                 categorical_imputation= 'mode',feature_selection = True,feature_selection_threshold= 0.8,ignore_low_variance = True, combine_rare_levels =True,

                                 remove_outliers= True,outliers_threshold = 0.01,feature_interaction = False, feature_ratio = False,silent = True,normalize = True, 

                                 normalize_method = 'zscore', transform_target = True, transform_target_method = 'yeo-johnson')
compare_models(blacklist =['tr'],turbo = True)
catboost = tune_model('catboost', n_iter = 50)

xgboost = tune_model('xgboost', n_iter = 50)

gbr = tune_model('gbr', n_iter = 50)

rf = tune_model('rf', n_iter = 100)

lightgbm = tune_model('lightgbm', n_iter = 50)

et = tune_model('et', n_iter = 100)

svm = tune_model('svm', n_iter = 100)
blend_specific = blend_models([catboost,xgboost,gbr,rf,lightgbm,et,svm] )
final_blender = finalize_model(blend_specific)
plot_model(rf,'feature') #'learning','vc','error'
predictions = predict_model(final_blender, data = test)
datasets=pd.concat([test['Id'],predictions['Label']],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission.csv',index=False)