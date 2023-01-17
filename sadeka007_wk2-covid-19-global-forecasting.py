import os

import math

from pprint import pprint



import pandas as pd

import numpy as np

import scipy as sp

import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt

import plotly



from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer



import lightgbm as lgb

from lightgbm import LGBMRegressor



from hyperopt import hp, tpe

from hyperopt.fmin import fmin
dirname = "/kaggle/input/train-with-containment-measures-taken/"

train_filename = "train-enriched-with-containment.csv"

train_new = "train.csv"

test_filename = "test.csv"

df_train_old = pd.read_csv(os.path.join(dirname, train_filename))

df_train_new = pd.read_csv(os.path.join("/kaggle/input/covid19-global-forecasting-week-2/", train_new))

df_test_raw = pd.read_csv(os.path.join("/kaggle/input/covid19-global-forecasting-week-2/", test_filename))

df_train_old.shape
df_train_new.shape
df_train_new["Date"].unique()
df_train_new["Province_State"].fillna(df_train_new["Country_Region"], inplace=True)
df_train_delta = df_train_new[~df_train_new["Date"].isin(df_train_old["Date"].values)]

df_train_delta.shape
df_train_delta["Date"].unique()
base_date = pd.to_datetime("2020-01-01")

df_train_delta.loc[:, "days_since"] = (pd.to_datetime(df_train_delta["Date"]) - base_date).dt.days

df_train_delta.head()
## 'Country', 'State', 'medicare', 'social_measure', 'travel_measure', 'isolation_measure','biz_measure', 'awareness_measure', 'lockdown_measure'

df_train_delta.loc[:, 'Country'] = None

df_train_delta.loc[:, 'State'] = None

df_train_delta.loc[:, 'medicare'] = None

df_train_delta.loc[:, 'social_measure'] = None

df_train_delta.loc[:, 'travel_measure'] = None

df_train_delta.loc[:, 'isolation_measure'] = None

df_train_delta.loc[:, 'biz_measure'] = None

df_train_delta.loc[:, 'awareness_measure'] = None

df_train_delta.loc[:, 'lockdown_measure'] = None
df_train_raw = pd.concat([df_train_old, df_train_delta])

df_train_raw.shape
df_train_raw["Date"].unique()
df_train_clean = df_train_raw.drop(["Id"], axis=1)

df_test_clean = df_test_raw[~df_test_raw["Date"].isin(df_train_raw["Date"])]

df_test_clean = df_test_clean.drop(["ForecastId"], axis=1)

print("train shape: ", df_train_clean.shape)

print("test shape: ", df_test_clean.shape)
df_train_clean["ConfirmedCases"] = np.log1p(df_train_clean["ConfirmedCases"])

df_train_clean["Fatalities"] = np.log1p(df_train_clean["Fatalities"])

df_test_clean["ConfirmedCases"] = None

df_test_clean["Fatalities"] = None
df = pd.concat([df_train_clean, df_test_clean], sort=False).reset_index(drop=True)

print(df.shape)

print(df.columns.values)
df["Date"] = pd.to_datetime(df["Date"])
df["Province_State"].fillna(value = df["Country_Region"], inplace = True)
def get_trend(df, col):

  trend_col = "trend_{}".format(col)

  df[trend_col] = (df[col] - df.groupby(["Country_Region", "Province_State"])[col].shift(fill_value=-999)) / (df.groupby(["Country_Region", "Province_State"])[col].shift(fill_value=0) + 0.0001)

  # df.loc[df[trend_col] > 100, trend_col] = 0

  

  return df



def get_lagged_value(df, col, start, end):

  for lag in list(range(start, end)):

    lagged_col = "{}-{}D".format(col, lag)

    print(lagged_col)

    df[lagged_col] = df.groupby(["Country_Region", "Province_State"])[col].shift(lag, fill_value=0)



  return df



def get_trendline(x, y, order=1):

  coeffs = np.polyfit(x, y, order)

  slope = coeffs[-2]



  return float(slope)

df_lagged = df.copy(deep=True)

df_lagged = get_trend(df_lagged, "ConfirmedCases")

df_lagged.head()
df_lagged = get_trend(df_lagged, "Fatalities")

df_lagged.head()
df_lagged["F2C_ratio"] = df_lagged["Fatalities"] / (df_lagged["ConfirmedCases"] + 0.0001)

df_lagged.loc[(df_lagged["ConfirmedCases"] == 0) & (df_lagged["F2C_ratio"] > 100), "F2C_ratio"] = 0

df_lagged.head()
df_lagged = get_lagged_value(df_lagged, "ConfirmedCases", 1, 7)

df_lagged = get_lagged_value(df_lagged, "Fatalities", 1, 7)
df_lagged.loc[(df_lagged["ConfirmedCases-1D"] == 0) & (df_lagged["trend_ConfirmedCases"] > 100), "trend_ConfirmedCases"] = 0
df_lagged.loc[(df_lagged["Fatalities-1D"] == 0) & (df_lagged["trend_Fatalities"] > 100), "trend_Fatalities"] = 0
df_lagged = get_lagged_value(df_lagged, "trend_ConfirmedCases", 1, 7)

df_lagged = get_lagged_value(df_lagged, "trend_Fatalities", 1, 7)

df_lagged = get_lagged_value(df_lagged, "F2C_ratio", 1, 7)
base_date = pd.to_datetime("2020-01-01")

df_lagged.loc[:, "days_since"] = (df_lagged["Date"] - base_date).dt.days
df_lagged.loc[:,"Province_State"] = df_lagged["Province_State"].astype("category")

df_lagged.loc[:,"Country_Region"] = df_lagged["Country_Region"].astype("category")
df_lagged.drop(["Country", "State"], axis=1, inplace=True)
measures = ["medicare", "social_measure", "travel_measure", "isolation_measure",

            "biz_measure", "awareness_measure", "lockdown_measure"]



for col in measures:

  df_lagged[col].fillna(value=0, inplace=True)
df_lagged["rolling_mean_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=6).mean())

df_lagged["rolling_std_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=6).std())

df_lagged["rolling_median_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=6).median())
df_lagged["rolling_mean_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=6).mean())

df_lagged["rolling_std_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=6).std())

df_lagged["rolling_median_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=6).median())
df_lagged = get_lagged_value(df_lagged, "rolling_mean_confirmedcases", 1, 2)

df_lagged = get_lagged_value(df_lagged, "rolling_std_confirmedcases", 1, 2)

df_lagged = get_lagged_value(df_lagged, "rolling_median_confirmedcases", 1, 2)

df_lagged = get_lagged_value(df_lagged, "rolling_mean_fatalities", 1, 2)

df_lagged = get_lagged_value(df_lagged, "rolling_std_fatalities", 1, 2)

df_lagged = get_lagged_value(df_lagged, "rolling_median_fatalities", 1, 2)
cols_float = ["ConfirmedCases", "ConfirmedCases-1D", "ConfirmedCases-2D", "ConfirmedCases-3D",

              "ConfirmedCases-4D", "ConfirmedCases-5D", "ConfirmedCases-6D", 

              "trend_ConfirmedCases", "trend_ConfirmedCases-1D", "trend_ConfirmedCases-2D", "trend_ConfirmedCases-3D", 

              "trend_ConfirmedCases-4D", "trend_ConfirmedCases-5D", "trend_ConfirmedCases-6D",

              "Fatalities", "Fatalities-1D", "Fatalities-2D", "Fatalities-3D", 

              "Fatalities-4D", "Fatalities-5D", "Fatalities-6D",

              "trend_Fatalities", "trend_Fatalities-1D", "trend_Fatalities-2D", "trend_Fatalities-3D",

              "trend_Fatalities-4D", "trend_Fatalities-5D", "trend_Fatalities-6D",

              "F2C_ratio", "F2C_ratio-1D", "F2C_ratio-2D", "F2C_ratio-3D",

              "F2C_ratio-4D", "F2C_ratio-5D", "F2C_ratio-6D"]



for col in cols_float:

  df_lagged[col] = df_lagged[col].astype(float)
df_lagged["rolling_mean_confirmedcases-1D"].fillna(value=0, inplace=True)

df_lagged["rolling_std_confirmedcases-1D"].fillna(value=0, inplace=True)

df_lagged["rolling_median_confirmedcases-1D"].fillna(value=0, inplace=True)

df_lagged["rolling_mean_fatalities-1D"].fillna(value=0, inplace=True)

df_lagged["rolling_std_fatalities-1D"].fillna(value=0, inplace=True)

df_lagged["rolling_median_fatalities-1D"].fillna(value=0, inplace=True)
df_train_clean["Date"].unique()
df_train = df_lagged[df_lagged["Date"] <= '2020-03-18']

df_valid = df_lagged[(df_lagged["Date"] > '2020-03-18') & (df_lagged["Date"] <= '2020-03-31')]

df_valid.tail(5)
start = df_lagged["days_since"].min()

end = df_lagged["days_since"].max()



for key, group in df_lagged.groupby(["Country_Region", "Province_State"]):

  prev = -1

  for i, row in group.iterrows():

    if prev == -1:

      prev = i

      continue

    else:

      for col in measures:

        prev_value = df_lagged.at[prev, col]



        if prev_value > 0:

          df_lagged.at[i, col] = prev_value



      prev = i
for col in measures:

  df_lagged.loc[:, col] = (df_lagged.loc[:, col] > 0) * 1
for key, group in df_lagged.groupby(["Country_Region", "Province_State"]):

  length = len(group[~group["ConfirmedCases"].isna()])

  x = range(0, length)

  yc = group[~group["ConfirmedCases"].isna()]["ConfirmedCases"]

  yf = group[~group["Fatalities"].isna()]["Fatalities"]



  for order in list(range(1, 6)):

    cc_slope = get_trendline(x, yc, order)

    f_slope = get_trendline(x, yf, order)



    for i, row in group.iterrows():

      df_lagged.at[i, "cc_slope_o{}".format(order)] = cc_slope

      df_lagged.at[i, "f_slope_o{}".format(order)] = f_slope
def rmsle(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    return np.sqrt(np.mean(np.power((y_pred - y_true), 2)))
df_train = df_lagged[df_lagged["Date"] <= '2020-03-18']

df_valid = df_lagged[(df_lagged["Date"] > '2020-03-18') & (df_lagged["Date"] <= '2020-03-31')]

df_valid.tail(5)
target = "ConfirmedCases"

droppable = ["ConfirmedCases", "Fatalities", "trend_ConfirmedCases", "trend_Fatalities", "F2C_ratio", 

             'rolling_mean_confirmedcases', 'rolling_std_confirmedcases', 'rolling_median_confirmedcases',

             'rolling_mean_fatalities', 'rolling_std_fatalities', 'rolling_median_fatalities']

y_train = df_train[target]

X_train = df_train.drop(droppable, axis = 1)

y_valid = df_valid[target]

X_valid = df_valid.drop(droppable, axis = 1)
tscv = TimeSeriesSplit(n_splits=56)

tscv
mse_scorer = make_scorer(rmsle, greater_is_better=True, needs_proba=False)
X_train.drop(["Date"], axis=1, inplace=True)

X_valid.drop(["Date"], axis=1, inplace=True)
len(X_train.columns.values)
# RMSE 0.238 params {'num_leaves': 10, 'colsample_bytree': '0.991', 'subsample_for_bin': 18000, 'reg_alpha': '0.224', 'reg_lambda': '0.972', 'min_child_samples': 15}

model_ConfirmedCases = LGBMRegressor(

        n_estimators=500,

        learning_rate=0.005,

        num_leaves=10,

        colsample_bytree=0.9913723835569889,

        subsample_for_bin=18000,

        reg_alpha=0.22430774790232594,

        reg_lambda=0.9715739602446141,

        min_child_samples=15,

        verbose=-1,

        n_jobs=-1,

        random_seed=42

        )



model_ConfirmedCases.fit(X_train, y_train)

# model_ConfirmedCases.booster_.save_model(os.path.join(dirname, "model_confirmed_cases.txt"))
y_pred_confirmedCases = model_ConfirmedCases.predict(X_valid)

print("test score: {:.3f}".format(rmsle(y_valid, y_pred_confirmedCases)))
df_allTrain = pd.concat([df_train, df_valid])

df_allTrain.columns.values
y = pd.concat([y_train, y_valid])

X = pd.concat([X_train, X_valid])

model_ConfirmedCases.fit(X, y)

# model_ConfirmedCases.booster_.save_model(os.path.join(dirname, "model_confirmed_cases.txt"))
len(X.columns.values)
target_1 = "Fatalities"



y_train = df_train[target_1]

y_valid = df_valid[target_1]
len(X_valid.columns.values)
model_Fatalities = LGBMRegressor(

        n_estimators=500,

        learning_rate=0.005,

        num_leaves=30,

        colsample_bytree=0.3723745674996714,

        subsample_for_bin=18000,

        reg_alpha=0.34560186656621394,

        reg_lambda=0.4249751776670454,

        min_child_samples=55,

        verbose=-1,

        n_jobs=-1,

        random_seed=42

        )



model_Fatalities.fit(X_train, y_train)

# model_Fatalities.booster_.save_model(os.path.join(dirname, "model_fatalities.txt"))
len(X_valid.columns.values)
y_pred_fatalities = model_Fatalities.predict(X_valid)

print("test score: {:.3f}".format(rmsle(y_valid, y_pred_fatalities)))
y = pd.concat([y_train, y_valid])

model_Fatalities.fit(X, y)
len(X.columns.values)
cutoff = (pd.to_datetime("2020-03-19") - base_date).days

start = (pd.to_datetime("2020-04-01") - base_date).days

end = (pd.to_datetime("2020-04-30") - base_date).days

print(start)

print(end)

print(cutoff)
df_test.columns.values
df_test = df_lagged[df_lagged["days_since"] >= cutoff].copy(deep=True)

df_test.drop([  'trend_ConfirmedCases'

              , 'trend_Fatalities'

              , 'F2C_ratio'

              , 'Date'

              , 'rolling_mean_confirmedcases'

              , 'rolling_std_confirmedcases'

              , 'rolling_median_confirmedcases'

              , 'rolling_mean_fatalities'

              , 'rolling_std_fatalities'

              , 'rolling_median_fatalities',], axis=1, inplace=True)
df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] < start), target] = y_pred_confirmedCases

df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] < start), target_1] = y_pred_fatalities
day = start



while day <= end: 

  df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] < start), target] = y_pred_confirmedCases

  df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] < start), target_1] = y_pred_fatalities



  df_test = get_trend(df_test, target)

  df_test = get_trend(df_test, target_1)

  df_test["F2C_ratio"] = df_test[target_1] / (df_test[target] + 0.0001)



  df_test.loc[(df_test[target] == 0) & (df_test["F2C_ratio"] > 100), "F2C_ratio"] = 0



  df_test = get_lagged_value(df_test, target, 1, 7)

  df_test = get_lagged_value(df_test, target_1, 1, 7)



  df_test.loc[(df_test[target] == 0) & (df_test["trend_ConfirmedCases"] > 100), "trend_ConfirmedCases"] = 0

  df_test.loc[(df_test[target] > 0) & (df_test["trend_ConfirmedCases"] > 100), "trend_ConfirmedCases"] = 1

  df_test.loc[(df_test[target_1] == 0) & (df_test["trend_Fatalities"] > 100), "trend_Fatalities"] = 0

  df_test.loc[(df_test[target_1] > 0) & (df_test["trend_Fatalities"] > 100), "trend_Fatalities"] = 1



  df_test = get_lagged_value(df_test, "trend_ConfirmedCases", 1, 7)

  df_test = get_lagged_value(df_test, "trend_Fatalities", 1, 7)

  df_test = get_lagged_value(df_test, "F2C_ratio", 1, 7)



  df_test["rolling_mean_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=6).mean())

  df_test["rolling_std_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=6).std())

  df_test["rolling_median_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=6).median())



  df_test["rolling_mean_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=6).mean())

  df_test["rolling_std_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=6).std())

  df_test["rolling_median_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=6).median())



  df_test = get_lagged_value(df_test, "rolling_mean_confirmedcases", 1, 2)

  df_test = get_lagged_value(df_test, "rolling_std_confirmedcases", 1, 2)

  df_test = get_lagged_value(df_test, "rolling_median_confirmedcases", 1, 2)

  df_test = get_lagged_value(df_test, "rolling_mean_fatalities", 1, 2)

  df_test = get_lagged_value(df_test, "rolling_mean_fatalities", 1, 2)

  df_test = get_lagged_value(df_test, "rolling_mean_fatalities", 1, 2)



  df_test["rolling_mean_confirmedcases-1D"].fillna(value=0, inplace=True)

  df_test["rolling_std_confirmedcases-1D"].fillna(value=0, inplace=True)

  df_test["rolling_median_confirmedcases-1D"].fillna(value=0, inplace=True)

  df_test["rolling_mean_fatalities-1D"].fillna(value=0, inplace=True)

  df_test["rolling_std_fatalities-1D"].fillna(value=0, inplace=True)

  df_test["rolling_median_fatalities-1D"].fillna(value=0, inplace=True)



  X = df_test.drop([target, target_1, "trend_ConfirmedCases", "trend_Fatalities", "F2C_ratio",

                   'rolling_mean_confirmedcases', 'rolling_std_confirmedcases', 'rolling_median_confirmedcases',

                   'rolling_mean_fatalities', 'rolling_std_fatalities', 'rolling_median_fatalities'], axis = 1)

  # print(X.isna().sum())



  df_test.loc[:, target] = model_ConfirmedCases.predict(X)

  df_test.loc[:, target_1] = model_Fatalities.predict(X)



  day = day + 1
df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] < start), target] = y_pred_confirmedCases

df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] < start), target_1] = y_pred_fatalities
df_test["ConfirmedCases"] = np.expm1(df_test["ConfirmedCases"])

df_test["Fatalities"] = np.expm1(df_test["Fatalities"])
df_temp = df_test_raw.copy(deep=True)

df_temp["Province_State"].fillna(value = df_temp["Country_Region"], inplace=True)



df_temp["Date"] = pd.to_datetime(df_temp["Date"])

df_temp["days_since"] = (df_temp["Date"] - base_date).dt.days

df_temp = pd.merge(df_temp, df_test, how="left", on=["days_since", "Province_State", "Country_Region"])

df_temp.shape
df_temp.isna().sum()
df_temp.loc[:, target] = np.ceil(df_temp[target]).astype(int)

df_temp.loc[:, target_1] = np.ceil(df_temp[target_1]).astype(int)
df_temp.describe()
df_temp = df_temp[["ForecastId", target, target_1]]

df_temp.to_csv("submission.csv", index=False)
len(df_temp[df_temp[target] < df_temp[target_1]])