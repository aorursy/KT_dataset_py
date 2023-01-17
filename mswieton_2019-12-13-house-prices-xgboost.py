# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import mean_squared_error, mean_squared_log_error
# checking xgboost version available for the kernel



#import xgboost

#xgboost.__version__
from xgboost import XGBRegressor



from xgboost import plot_importance
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col=0)

train_data
train_cols = train_data.columns.to_list()

train_cols
# setting prediction target



tgt_col = train_cols[-1]

print('Prediction target name: {}'.format(tgt_col))
train_data.info()
train_data.describe()
# checking features with most missing values ("train" dataset)



train_data.isna().sum().sort_values(ascending=False)
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col=0)

test_data
test_cols = test_data.columns.to_list()

test_cols
test_data.info()
test_data.describe()
# checking features with most missing values ("test" dataset)



test_data.isna().sum().sort_values(ascending=False)
sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv', index_col='Id')

sample_submission
# setting the value to replace all missing feature values



missing = -999
# preparing train data by dropping target variable and filling missing values



X_train = train_data.drop(columns=[tgt_col])

X_train = X_train.fillna(missing)

X_train.shape
# preparing prediction target

y_train = train_data[tgt_col]



# checking if there are no missing target values

y_train.isna().sum() == 0
# preparing test data by filling missing values



X_test = test_data.copy()

X_test = X_test.fillna(missing)

X_test.shape
# checking if feature columns match for train and test data



X_train.columns == X_test.columns
# deleting dataset no longer necessary



del train_data, test_data
# Label Encoding for train&test explanatory variables



for col in test_cols:

    if X_train[col].dtype=='object' or X_test[col].dtype=='object':

        print('Encoding categorical feature: {}'.format(col))

        le = LabelEncoder()

        le.fit(list(X_train[col].values) + list(X_test[col].values))

        X_train[col] = le.transform(list(X_train[col].values))

        X_test[col] = le.transform(list(X_test[col].values))
"""helper function to show feature importances for a model provided"""



def show_feat_importances(model):

    

    # getting feature importances

    imp_type = model.importance_type

    feat_imp = pd.DataFrame(model.feature_importances_, 

                            columns = ['importance'], index=test_cols)

    feat_imp.sort_values(by='importance', axis=0, ascending=False, inplace=True)

    

    print('\nFeature importances (type: {}) for this model: '

          .format(imp_type))    

    print(feat_imp)

    

    # plotting feature importances    

    plot = plot_importance(model, importance_type='gain', 

                           title='Feature importance by gain')

    plot.figure.set_size_inches(12,16)

    plt.show()



    plot = plot_importance(model, importance_type='weight', 

                           title='Feature importance by weight')

    plot.figure.set_size_inches(12,16)

    plt.show()

    

    return feat_imp
# maximum tree depth

max_depth = 3



# number of trees to fit

n_estimators = 1000



# setting the learning task and the corresponding learning objective

"""NOTE: accoring to evaluation metric: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/evaluation,

objective = 'reg:squaredlogerror' should be used, but it is not supported in xgboost v. 0.90 - 

we get: XGBoostError: Unknown objective function reg:squaredlogerror

Therefore we use:"""

objective = 'reg:squarederror'



# learning rate

learning_rate = 0.10



# setting the feature importance type for the model's feature_importances_ property

importance_type='gain'
model_XGBR = XGBRegressor(booster='gbtree', 

                          importance_type=importance_type,

                          learning_rate=learning_rate,

                          max_depth=max_depth, 

                          missing=missing, 

                          n_estimators=n_estimators, 

                          n_jobs=-1, 

                          objective=objective,

                          random_state=42, 

                          verbosity=1

                         )

model_XGBR
# fitting model on full data



model_XGBR.fit(X_train, y_train)
# making predictions for train dataset (i.e. in-sample check)

y_pred_train = model_XGBR.predict(X_train)



# calculating error metrics for in-sample check

MSE = mean_squared_error(y_true=y_train, y_pred=y_pred_train)

MSLE = mean_squared_log_error(y_true=y_train, y_pred=y_pred_train)

print('\nError metrics on full train data:')

print('MSE: {}'.format(MSE))

print('MSLE: {}'.format(MSLE))



# showing feature importances

show_feat_importances(model=model_XGBR)
# making predictions on test data

y_pred_test = model_XGBR.predict(X_test)

print('\nGenerated {} test predictions'.format(y_pred_test.shape[0]))    

    

sample_submission['SalePrice'] = y_pred_test
sample_submission.to_csv('submission.csv')