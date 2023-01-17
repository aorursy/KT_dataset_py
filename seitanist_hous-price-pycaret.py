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
        
        
!pip install pycaret
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pycaret.regression import *
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.info()
clf_setup = setup(data = train, target = "SalePrice", session_id = 1)
compare_models(fold = 5, sort = "RMSE")
catboost = create_model("catboost")
tuned_catboost = tune_model(catboost)
interpret_model(catboost)
interpret_model(catboost, plot = "reason")
interpret_model(catboost, plot = "correlation")
pred = predict_model(catboost)
final_catboost = finalize_model(catboost)
final_pred = predict_model(final_catboost, data = test)
final_pred.head()
holdout_id = test.Id
submit = {"Id":holdout_id, "SalePrice":final_pred.Label}
submit_df = pd.DataFrame(submit)
submit_df.head()
submit_df.to_csv("submission_pycaret_simple_catboost.csv", index=False)
