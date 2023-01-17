# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install pycaret
from pycaret.regression import *
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub   = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

print('Training Data = ',train.shape)
print('Testing Data = ',test.shape)
train.head(3)
def impute_missing_values(df):
    
    # Get Continuous and Categorical Features
    missing_cols = df.columns[df.isna().sum()>0]
    mean_cols = df[missing_cols].describe().columns
    mode_cols = list((set(missing_cols) - set(mean_cols)))

    # Impute Missing Values
    for col in mean_cols:
        df[col].fillna(df[col].mean(),axis=0,inplace=True)    

    for col in mode_cols:
        df[col].fillna('Unknown',axis=0,inplace=True)
        
    return df

train = impute_missing_values(train)
test  = impute_missing_values(test)

print('\nTraining Set after Imputation\nShape=',train.shape,'\nMissing Values=',train.isna().sum().sum())    
print('\nTesting Set after Imputation\nShape=',test.shape,'\nMissing Values=',test.isna().sum().sum())    
A = set(train.columns)
B = set(test.columns)

print('Uncommon Columns are ',A.union(B) - A.intersection(B))
clf1 = setup(data       = train, 
             target     = 'SalePrice',             
             session_id = 123)
compare_models(sort='MSE', fold=2)
cat = create_model('catboost',fold=10)
tune_cat = tune_model('catboost',fold=10,optimize='mse')
interpret_model(cat)
interpret_model(tune_cat)
xgb = create_model('xgboost',fold=10)
tune_xgb = tune_model('xgboost',fold=10,optimize='mse')
plot_model(xgb,plot='residuals')
plot_model(xgb,plot='error')
plot_model(xgb,plot='feature')
plot_model(tune_xgb,plot='residuals')
plot_model(tune_xgb, plot='error')
plot_model(tune_xgb,plot='feature')
rf = create_model('rf',fold=10)
tune_rf = tune_model('rf',fold=10,optimize='mse')
plot_model(rf, plot='residuals')
plot_model(tune_rf, plot='residuals')
plot_model(rf, plot='error')
plot_model(tune_rf, plot='error')
plot_model(rf, plot='feature')
plot_model(tune_rf, plot='feature')
xgb_pred = predict_model(xgb, data=test)
xgb_pred.head(3)

tune_cat_pred = predict_model(tune_cat, data=test)
tune_cat_pred.head(3)

sub1 = xgb_pred[['Id','Label']]
sub1.columns = ['Id','SalePrice']
sub1.to_csv('submission_xgb.csv',index=False)
sub1.head()
sub2 = tune_cat_pred[['Id','Label']]
sub2.columns = ['Id','SalePrice']
sub2.to_csv('submission_tune_cat.csv',index=False)
sub2.head()
