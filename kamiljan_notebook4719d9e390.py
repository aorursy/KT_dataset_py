# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/covid19-global-forecasting-week-5/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-5/test.csv")
train_df.shape, test_df.shape
train_df.head()
train_df.isnull().sum()
test_df.isnull().sum()
train_df.drop(["County", "Target"], axis=1, inplace=True)
test_df.drop(["County", "Target"], axis=1, inplace=True)
train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

train_df['day'] = train_df['Date'].dt.day
train_df['month'] = train_df['Date'].dt.month
train_df['dayofweek'] = train_df['Date'].dt.dayofweek
train_df['dayofyear'] = train_df['Date'].dt.dayofyear
train_df['quarter'] = train_df['Date'].dt.quarter
train_df['weekofyear'] = train_df['Date'].dt.weekofyear

test_df['day'] = test_df['Date'].dt.day
test_df['month'] = test_df['Date'].dt.month
test_df['dayofweek'] = test_df['Date'].dt.dayofweek
test_df['dayofyear'] = test_df['Date'].dt.dayofyear
test_df['quarter'] = test_df['Date'].dt.quarter
test_df['weekofyear'] = test_df['Date'].dt.weekofyear
train_df.drop("Date", axis=1, inplace=True)
test_df.drop("Date", axis=1, inplace=True)
train_df.head()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

train_df.fillna("Nan", inplace=True)
test_df.fillna("Nan", inplace=True)

oe = OrdinalEncoder()
train_df[["Country_Region", "Province_State"]] = oe.fit_transform(train_df[["Country_Region", "Province_State"]])
test_df[["Country_Region", "Province_State"]] = oe.fit_transform(test_df[["Country_Region", "Province_State"]])
train_df.drop("Id", axis=1, inplace=True)
train_df.head()
test_df.drop("ForecastId", axis=1, inplace=True)
test_df.head()
def get_train_valid_split(df):
    last_month = df["month"].max()
    X_train = df[df["month"] < last_month]
    y_train = X_train["TargetValue"].values
    X_train.drop("TargetValue", axis=1, inplace=True)
    
    X_valid = df[df["month"] == last_month]
    y_valid = X_valid["TargetValue"].values
    X_valid.drop("TargetValue", axis=1, inplace=True)
    return X_train, y_train, X_valid, y_valid
X_train, y_train, X_valid, y_valid = get_train_valid_split(train_df)
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_train.columns = train_df.columns.drop("TargetValue")
X_valid = pd.DataFrame(scaler.transform(X_valid))
X_valid.columns = train_df.columns.drop("TargetValue")
# n_estimators_list = [100, 300, 500, 1000, 1200]
# learning_rate_list = [0.01, 0.05, 0.1, 0.3]

# best_score = float("inf")
# best_model = -1
# best_ne = -1
# best_lr = -1
# for n_estimators in n_estimators_list:
#     print(f"\nn_estimators: {n_estimators}\n")
#     for learning_rate in learning_rate_list:
#         print(f"learning_rate: {learning_rate}", end=" ")
#         my_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=4, seed=0)
#         my_model.fit(X_train, y_train,
#                      early_stopping_rounds=10,
#                      eval_set=[(X_valid, y_valid)],
#                      verbose=False)
#         score = mean_absolute_error(y_valid, my_model.predict(X_valid))
#         print(score)
#         if score < best_score:
#             best_model = my_model
#             best_ne = n_estimators
#             best_lr = learning_rate
#             best_score = score

# print(f"Best n_estimators: {best_ne}")
# print(f"Best learning_rate: {best_lr}")
my_model = XGBRegressor(n_estimators=100, learning_rate=0.3, max_depth=15, n_jobs=4, seed=0)
my_model.fit(X_train, y_train,
             early_stopping_rounds=10, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
score = mean_absolute_error(y_valid, my_model.predict(X_valid))
print(score)
X_test = pd.DataFrame(scaler.transform(test_df))
X_test.columns = test_df.columns
preds = my_model.predict(X_test)
preds = np.around(preds)
preds
output = pd.DataFrame({'Id': test_df.index + 1, 'TargetValue': preds})
output.head()
a = output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b = output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c = output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns = ['Id', 'q0.05']
b.columns = ['Id', 'q0.5']
c.columns = ['Id', 'q0.95']
a = pd.concat([a, b['q0.5'], c['q0.95']], axis=1)
a.head()
sub = pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable'] = sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile'] = sub['Id'].astype(str) + '_' + sub['variable']
sub['TargetValue'] = sub['value']
sub=sub[['ForecastId_Quantile', 'TargetValue']]
sub.reset_index(drop=True, inplace=True)
sub.to_csv("submission.csv", index=False)
sub.head()
from xgboost import plot_importance

_, ax = plt.subplots(figsize=(7, 7))
plot_importance(my_model, ax=ax)
plt.show()