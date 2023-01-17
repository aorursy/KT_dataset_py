# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train= pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test= pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train.head()
train.info()
train.columns
train["SalePrice"].describe()
print(train.isnull().sum().sort_values(ascending= False).head(20))
!pip install pycaret
from pycaret.regression import *
PPL= setup(data= train, target= "SalePrice", numeric_imputation= "mean", categorical_imputation= "constant")
compare_models()
LLAR= create_model("lasso")
plot_model(LLAR)
evaluate_model(LLAR)
#Predictions on hold-out set
LLAR_pred_holdout = predict_model(LLAR)
#Predictions on new dataset
prediction_test = predict_model(LLAR, data=test)
prediction_test.head()
sub= pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
sub["SalePrice"]= round(prediction_test["Label"].astype(int))
sub.to_csv("submission.csv", index=False)
sub.head()
                         
sub["SalePrice"]= prediction_test["Label"]
sub.to_csv("submission_house_price.csv", index= False)
sub.head()