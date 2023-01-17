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
!pip install pycaret
import pandas as pd

titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_train.head()
titanic_test = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_test.head()
# Submission File

sub = pd.read_csv('../input/titanic/gender_submission.csv')
# Survived is a binary column hence imported classification module

from pycaret.classification import* 



clf1 = setup(data = titanic_train, ignore_features = ['PassengerId', 'Name', 'Ticket'], target = 'Survived')
# comparing all models

compare_models()
# Tune the best performing model - LightGBM

tuned_lightgbm = tune_model('lightgbm')
# Tune the best performing model - LightGBM

tuned_lightgbm_auc = tune_model('lightgbm', optimize = 'AUC')
# Tune the best performing model - XGBoost

tune_xgb_titanic = tune_model('xgboost', n_iter = 100, )
# evaluate_model(tune_xgb_titanic)

evaluate_model(tuned_lightgbm_auc)
# final_xgboost = finalize_model(tune_xgb_titanic)

final_lgbm = finalize_model(tuned_lightgbm_auc)
# print(final_xgboost)

print(final_lgbm)
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')

titanic_test.head()
# predict on hold-out - 

# pred_holdout = predict_model(final_xgboost)

pred_holdout = predict_model(final_lgbm)
# predict on TEST data

# pred_test_data = predict_model(final_xgboost, data = titanic_test)

pred_test_data = predict_model(final_lgbm, data = titanic_test)
pred_test_data.head()
!pwd
sub['Survived'] = round(pred_test_data['Score']).astype(int)

sub.to_csv('submission.csv',index=False)

sub.head()