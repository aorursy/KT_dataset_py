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
train_df = pd.read_csv("/kaggle/input/Train.csv")

train_df.head(2)
test_df = pd.read_csv("/kaggle/input/Test.csv")

test_df.head(2)
train_df.info()
train_df = train_df.dropna()
new_train_df = train_df.drop(['Age', 'Attrition_rate', 'Employee_ID'], axis=1)

new_train_df = pd.get_dummies(new_train_df)

new_train_df
test_df.info()
new_test_df = test_df.drop(['Age', 'Employee_ID'], axis=1)

new_test_df = pd.get_dummies(new_test_df)

new_test_df
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')

imputer.fit(new_test_df)

n_test_df = pd.DataFrame(imputer.transform(new_test_df))

n_test_df.columns=new_test_df.columns

n_test_df.index=new_test_df.index



n_test_df.info()
# For readibility, putting the training feature and target into X and y

X = new_train_df

y = train_df['Attrition_rate']
# Train Test Split.

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Model training and hyperparameter tuning.



from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV



parameters = {"learning_rate": [0.1, 0.01, 0.001],

               "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],

               "max_depth": [2, 4, 7, 10],

               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],

               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],

               "reg_alpha": [0, 0.5, 1],

               "reg_lambda": [1, 1.5, 2, 3, 4.5],

               "min_child_weight": [1, 3, 5, 7],

               "n_estimators": [100, 250, 500, 1000]}



xgb_rscv = RandomizedSearchCV(XGBRegressor(), param_distributions = parameters, scoring = "neg_root_mean_squared_error",

                             cv = 7, verbose = 3, random_state = 40)



model_rscv = xgb_rscv.fit(X_train, y_train)

model_rscv.best_params_
model_xgboost = XGBRegressor(booster='gbtree', subsample=0.6,

 reg_lambda=1,

 reg_alpha=0,

 n_estimators=100,

 min_child_weight=5,

 max_depth=2,

 learning_rate=0.1,

 gamma=1.5,

 colsample_bytree=1.0)



model_xgboost.fit(X_train, y_train)
# Scoring (Root mean squared error : The lesser the better)



from sklearn.metrics import mean_squared_error



pred = model_xgboost.predict(X_test)



rmse = np.sqrt(mean_squared_error(y_test, pred))

print("Error (the lesser the better): %f" % (rmse))
# Generating Submission file.

prediction = model_xgboost.predict(n_test_df)



output = pd.DataFrame({'Employee_ID': test_df.Employee_ID, 'Attrition_rate': prediction})

output.to_csv('my_submission.csv', index=False)