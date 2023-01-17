# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv", parse_dates=[2])
train_df.describe()
# Any results you write to the current directory are saved as output.
def load_raw_df():
    train_df = pd.read_csv("../input/train.csv", parse_dates=[2])
    store_df = pd.read_csv("../input/store.csv")
    test_df = pd.read_csv("../input/test.csv", parse_dates=[3])
    train_df.head()
    store_df.head()
    test_df.head()
    display(train_df.isnull().sum(),store_df.isnull().sum(),test_df.isnull().sum())
    mereged_train_df = pd.merge(train_df, store_df, on='Store')
    merged_test_df = pd.merge(test_df, store_df, on='Store')
    return mereged_train_df, merged_test_df

def extract_linear_feature(df):
    df.drop("Customers", 1)
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    df["StoreType"] = df["StoreType"].map(mappings)
    df["Assortment"] = df["Assortment"].map(mappings)
    df["StateHoliday"].loc[df["StateHoliday"] == 0] = "0"
    df["StateHoliday"] = df["StateHoliday"].map(mappings)
    df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    # merege_train_df['WeekOfYear'] = merege_train_df.Date.dt.weekofyear
#     df['store_a'] = df['StoreType'] == 3
    df["year"] = df.Date.dt.year
    df["month"] = df.Date.dt.month
    df["day"] = df.Date.dt.day
    df["year"] = df.Date.dt.year
    df["month"] = df.Date.dt.month
    df["day"] = df.Date.dt.day
    
    df.drop(["Date", "PromoInterval", "Promo2SinceWeek", "Promo2SinceYear"], axis=1, inplace=True)
    return df

# return x_train, y_train, x_test
def get_train_test_label():
    train_df, test_df = load_raw_df()
    train_df_2 = extract_linear_feature(train_df)
    train_df_2.head()
    y_true = train_df_2.loc[:, 'Sales'].as_matrix(columns=None)
    train_df_2.drop("Sales", axis =1)

    test_df["Customers"] = 1
    test_df = extract_linear_feature(test_df)
    test_df.head()
    feature_list = list(train_df_2.columns)
    print(feature_list)
    return train_df_2,y_true,test_df

x_train, y_train, x_test = get_train_test_label()
# features = ['Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'year', 'month', 'day']


# from sklearn.linear_model import LinearRegression
# x_train.fillna(0)
# # x_train = x_train[x_train["Open"] != 0]
# x_test.fillna(0)
# features = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'year', 'month', 'day']
# features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment','Promo2', 'year', 'month', 'day']

# lm_reg = LinearRegression()
# lm_reg.fit(x_train[features].values, list(y_train))
# x_test_closed = x_test["Id"][x_test["Open"] == 0].values
# x_test = x_test[x_test["Open"] != 0]

# y_test_pred = lm_reg.predict(x_test[features])
# print(y_test_pred[:10])

# result = pd.Series()
# result = result.append(pd.Series(y_test_pred, index=x_test["Id"]))
# result = result.append(pd.Series(0, index=x_test_closed))
# result = pd.DataFrame({ "Id": result.index, "Sales": result.values})
# result.to_csv('result_new.csv', index=False)

# print(os.listdir("."))
import xgboost as xgb
from sklearn.cross_validation import train_test_split
train_df = pd.read_csv("../input/train.csv", parse_dates=[2])
store_df = pd.read_csv("../input/store.csv")
test_df = pd.read_csv("../input/test.csv", parse_dates=[3])
features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment','Promo2', 'WeekOfYear','year', 'month', 'day']
def extract_feature(df):
#     df.drop("Customers", 1)
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    df["StoreType"] = df["StoreType"].map(mappings)
    df["Assortment"] = df["Assortment"].map(mappings)
    df["StateHoliday"].loc[df["StateHoliday"] == 0] = "0"
    df["StateHoliday"] = df["StateHoliday"].map(mappings)
    df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df['WeekOfYear'] = df.Date.dt.weekofyear
    df["year"] = df.Date.dt.year
    df["month"] = df.Date.dt.month
    df["day"] = df.Date.dt.day
#     df.drop(["Date", "PromoInterval", "Promo2SinceWeek", "Promo2SinceYear"], axis=1, inplace=True)
    return df

train_df = pd.merge(train_df, store_df, on='Store')
test_df = pd.merge(test_df, store_df, on='Store')
train_df = extract_feature(train_df)
test_df = extract_feature(test_df)
train_df = train_df[train_df["Open"] != 0]

train, valid = train_test_split(train_df, test_size=0.03)
y_train = np.log1p(train.Sales)
y_valid = np.log1p(valid.Sales)
dtrain = xgb.DMatrix(train[features], y_train)
dvalid = xgb.DMatrix(valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
def rmspe(y, yhat):
    return np.sqrt(np.mean(((y - yhat)/y) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.1,
          "max_depth": 10,
          "subsample": 0.85,
          "colsample_bytree": 0.4,
          "min_child_weight": 6,
          "thread": 1,
          "seed": 10
          }
num_boost_round = 1000
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=1000, \
  feval=rmspe_xg, verbose_eval=True)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(valid[features]))
error = rmspe(valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the test set")
dtest = xgb.DMatrix(test_df[features])
test_probs = gbm.predict(dtest)
print(test_probs[:5])
# Make Submission
result = pd.DataFrame({"Id": test_df["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv("xgb_v2.csv", index=False)


