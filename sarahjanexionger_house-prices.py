import numpy as np

from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb
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
train_data = pd.read_csv(os.path.join(dirname, 'train.csv'))
x_test = pd.read_csv(os.path.join(dirname, 'test.csv'))
ids = x_test.Id
y_train = train_data["SalePrice"]
x_train = train_data.drop('SalePrice', axis=1)

all_data = x_train
all_data = all_data.append(x_test)
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
all_data = all_data.drop("Id", axis=1)
x_train = all_data.iloc[:len(x_train), :]
x_test = all_data.iloc[len(x_train):, :]
print(x_train.info())
print(x_test.info())

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", colsample_bytree=0.9820400327373335, gamma=0.48683191837765866, learning_rate=0.11517629242123971, subsample=0.7942455014344907, n_estimators=117, max_depth=3, random_state=42)
xgb_model.fit(x_train, y_train)
predictions = xgb_model.predict(x_test)
print(predictions)
predictions = predictions.reshape(-1)

output = pd.DataFrame({'Id': ids, 'SalePrice': predictions})
output.to_csv('xgb_submission.csv', index=False)

