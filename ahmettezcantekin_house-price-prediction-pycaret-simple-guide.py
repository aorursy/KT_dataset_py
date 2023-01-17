#Install pycaret library
#Make sure Internet option is on in the settings panel!
!pip install pycaret
#Importing the neccessary libraries!
import pandas as pd
from pycaret.regression import *
#Read the data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#pycaret regression setup

target='SalePrice'

setup_reg = setup(
    data = train,
    target = target,
    train_size=0.8,
    numeric_imputation = 'mean',
    silent = True,
    remove_outliers=True,
    normalize=True
)
#Comparing the models with blacklist option!
#You can also specify fold number, sorting metric etc!
#compare_models(blacklist = None, fold = 10,  round = 4,  sort = ‘R2’, turbo = True)

bl_models = ['ransac', 'tr', 'rf', 'et', 'ada', 'gbr']

result=compare_models(
    #blacklist = bl_models,
    fold = 5,
    sort = 'MAE', ## competition metric
    turbo = True
)
result
#You can check estimator options from https://pycaret.org/regression/
#This method returns MAE, MSE, RMSE, R2, RMSLE and MAPE values for selected models.
cb = create_model(
    estimator='catboost',
    fold=5
)
#Pycaret also has tune option for selected model!
#This function tunes the hyperparameters of a model and scores it using K-fold Cross Validation. 
tuned_cb = tune_model(
    cb,
    fold=5
)
#prediction
predictions =  predict_model(tuned_cb, data=test)
predictions.head()
#you can also ensemble your trained model
ensembled_cb = ensemble_model(cb)
#prediction
predictions_ensemble =  predict_model(ensembled_cb, data=test)
predictions_ensemble.head()
#prepare the submission file
predictions_ensemble.rename(columns={'Label':'SalePrice'}, inplace=True)
predictions_ensemble[['Id','SalePrice']].to_csv('submission_ensemble.csv', index=False)

#you can also blend the models
#Blend function creates an ensemble meta-estimator that fits a base regressor on the whole dataset. It then averages the predictions to form a final prediction.
#create models for blending
cb = create_model(estimator='catboost',fold=5)
par = create_model(estimator='par',fold=5)
hr = create_model(estimator='huber',fold=5)
#blend trained models
blend_specific = blend_models(estimator_list = [cb,par,hr])
#prediction
predictions_blended =  predict_model(blend_specific, data=test)
predictions_blended.head()
#prepare the submission file
predictions_blended.rename(columns={'Label':'SalePrice'}, inplace=True)
predictions_blended[['Id','SalePrice']].to_csv('submission_blended.csv', index=False)
