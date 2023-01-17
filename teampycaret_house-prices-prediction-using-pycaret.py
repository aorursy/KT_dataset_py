import pandas as pd

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head()
!pip install pycaret
from pycaret.regression import *

reg1 = setup(train, target = 'SalePrice', session_id = 123, silent = True) #silent is set to True for unattended run during kernel execution
compare_models(blacklist = ['tr']) #blacklisted Thielsen Regressor due to longer training times
catboost = create_model('catboost', verbose = False) #verbose set to False to avoid printing score grid

gbr = create_model('gbr', verbose = False)

xgboost = create_model('xgboost', verbose = False)
blend_top_3 = blend_models(estimator_list = [catboost, gbr, xgboost])
stack1 = stack_models(estimator_list = [gbr, xgboost], meta_model = catboost, restack = True)
from pycaret.regression import *

reg1 = setup(train, target = 'SalePrice', session_id = 123, 

             normalize = True, normalize_method = 'zscore',

             transformation = True, transformation_method = 'yeo-johnson', transform_target = True,

             ignore_low_variance = True, combine_rare_levels = True,

             numeric_features=['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 

                               'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 

                               'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'PoolArea'],

             silent = True #silent is set to True for unattended run during kernel execution

             )
compare_models(blacklist = ['tr']) #blacklisted Thielsen Regressor due to longer training times
gbr = create_model('gbr', verbose = False)

catboost = create_model('catboost', verbose = False)

svm = create_model('svm', verbose = False)

lightgbm = create_model('lightgbm', verbose = False)

xgboost = create_model('xgboost', verbose = False)
blend_top_5 = blend_models(estimator_list = [gbr,catboost,svm,lightgbm,xgboost])
stack2 = stack_models(estimator_list = [gbr,catboost,lightgbm,xgboost], meta_model = svm, restack = True)
from pycaret.regression import *

reg1 = setup(train, target = 'SalePrice', session_id = 123, 

             normalize = True, normalize_method = 'zscore',

             transformation = True, transformation_method = 'yeo-johnson', transform_target = True,

             numeric_features=['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 

                               'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 

                               'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'PoolArea'],

             ordinal_features= {'ExterQual': ['Fa', 'TA', 'Gd', 'Ex'],

                                'ExterCond' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'BsmtQual' : ['Fa', 'TA', 'Gd', 'Ex'], 

                                'BsmtCond' : ['Po', 'Fa', 'TA', 'Gd'],

                                'BsmtExposure' : ['No', 'Mn', 'Av', 'Gd'],

                                'HeatingQC' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'KitchenQual' : ['Fa', 'TA', 'Gd', 'Ex'],

                                'FireplaceQu' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'GarageQual' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'GarageCond' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'PoolQC' : ['Fa', 'Gd', 'Ex']},

             polynomial_features = True, trigonometry_features = True, remove_outliers = True, outliers_threshold = 0.01,

             silent = True #silent is set to True for unattended run during kernel execution

             )
compare_models(blacklist = ['tr']) #blacklisted Thielsen Regressor due to longer training times
huber = tune_model('huber', n_iter = 100)
omp = tune_model('omp', n_iter = 100)
ridge = tune_model('ridge', n_iter = 100)
br = tune_model('br', n_iter = 100)
lightgbm = tune_model('lightgbm', n_iter = 50)
par = tune_model('par', n_iter = 100)
blend_all = blend_models(estimator_list = [huber, omp, ridge, br])
plot_model(br, plot = 'residuals')
plot_model(br, plot = 'error')
plot_model(br, plot = 'vc')
plot_model(br, plot = 'feature')
interpret_model(lightgbm)
interpret_model(lightgbm, plot = 'correlation', feature = 'TotalBsmtSF')
interpret_model(lightgbm, plot = 'reason', observation = 0)
# check predictions on hold-out

predict_model(blend_all);
final_blender = finalize_model(blend_all)

print(final_blender)
predictions = predict_model(final_blender, data = test)

predictions.head()