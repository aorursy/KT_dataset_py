!pip install pycaret
from pycaret.regression import *
import pandas as pd
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.head()
reg = setup(data = train,target = 'SalePrice', numeric_imputation = 'mean',normalize = True,
             ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],pca=True,
    pca_method='linear',pca_components=10)
compare_models(exclude = ['tr'] , turbo = True) 
cb = create_model('catboost')
prediction = predict_model(cb, data = test)
output = pd.DataFrame({'Id': test.Id, 'SalePrice': prediction.Label})
output.to_csv('submission.csv', index=False)
output.head()
