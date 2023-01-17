# installing pycaret

!pip install pycaret
import pandas as pd

from pycaret.regression import*
#loading data

train = pd.read_csv('/content/train.csv')

test = pd.read_csv('/content/test.csv')
print(train.shape, test.shape)
train.head(3)
test.head(3)
#preparing the data for modeling

setup(train, target = 'SalePrice',remove_outliers= True, normalize= True, normalize_method = 'zscore' )
#selecting the best model

#FOLD = 10

compare_models(blacklist=['tr'])
catboost = create_model('catboost')
#Tunning hyperparameters

tuned_catboost = tune_model('catboost')
# ensembling a trained dt model

catboost_bagged = ensemble_model(catboost)
# predict test / hold-out dataset

holdout_pred = predict_model(catboost_bagged)

#final predicition

final_pred = predict_model(catboost_bagged, data= test)
final_pred.head()
y_pred = pd.DataFrame()

y_pred['Id'] = final_pred['Id']

y_pred['SalePrice'] = final_pred.Label

y_pred.to_csv('submission.csv',index=False)