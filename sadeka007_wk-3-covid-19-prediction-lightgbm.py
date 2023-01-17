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
df_train_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

df_train_raw.shape
df_env = pd.read_csv("/kaggle/input/environmental-variables-for-world-countries/World_countries_env_vars.csv")

df_env.shape
# df_train_raw[~df_train_raw["Country_Region"].isin(df_env["Country"].values)]["Country_Region"].unique()

df_train_raw["Country_Region"].unique()
df_env[~df_env["Country"].isin(df_train_raw["Country_Region"].unique())]["Country"].values
df_env.loc[~df_env["Country"].isin(df_train_raw["Country_Region"].unique()), "Country"] = df_env["Country"].map({"United States of America": "US",

                                                                                                                 "Democratic Republic of the Congo": "Congo (Kinshasa)",

                                                                                                                 "Myanmar": "Burma",

                                                                                                                 "French Polynesia": "France",

                                                                                                                 "South Sudan": "Sudan",

                                                                                                                 "United Republic of Tanzania": "Tanzania",

                                                                                                                 "Republic of the Congo": "Congo (Brazzaville)",

                                                                                                                 "North Korea": "Korea, North",

                                                                                                                 "South Korea": "Korea, South",

                                                                                                                 "Taiwan": "Taiwan*",

                                                                                                                 "Republic of Serbia": "Serbia",

                                                                                                                 "Czech Republic": "Czechia",

                                                                                                                 "French Guiana": "Guinea",

                                                                                                                 "Guinea Bissau": "Guinea-Bissau",

                                                                                                                 "Ivory Coast": "Cote d'Ivoire",

                                                                                                                 "Northern Cyprus": "Cyprus",

                                                                                                                 "Cape Verde": "Cabo Verde",

                                                                                                                 "East Timor": "Timor-Leste",

                                                                                                                 "The Bahamas": "Bahamas",

                                                                                                                 "West Bank": "West Bank and Gaza"

})
print("env data shape: {}".format(df_env.shape))

df_env = df_env[df_env["Country"].isin(df_train_raw["Country_Region"].values)]

print("env data shape: {}".format(df_env.shape))

df_env.columns.values
# df_env.isna().sum()

# df_env.to_csv("environment_worldwide.csv", index=False)
df_airpol = pd.read_csv("/kaggle/input/pm25-global-air-pollution-20102017/PM2.5 Global Air Pollution 2010-2017.csv")

df_airpol.shape
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
dirname = '/kaggle/input'

train_enriched_filename = "train-with-containment-measures-taken/train-enriched-with-containment.csv"

train_filename = "covid19-global-forecasting-week-3/train.csv"

test_filename = "covid19-global-forecasting-week-3/test.csv"

df_train_enriched = pd.read_csv(os.path.join(dirname, train_enriched_filename))

df_train_raw = pd.read_csv(os.path.join(dirname, train_filename))

df_test_raw = pd.read_csv(os.path.join(dirname, test_filename))

df_train_raw.shape
df_airpol.loc[~df_airpol["Country Name"].isin(df_train_raw["Country_Region"].values), "Country Name"] = df_airpol["Country Name"].map({"American Samoa": "Samoa","Arab World": "Saudi Arabia", "Bahamas, The": "Bahamas", "Myanmar": "Burma", "Brunei Darussalam": "Brunei", "Congo, Dem. Rep.": "Congo (Kinshasa)", 

 "Congo, Rep.": "Congo (Brazzaville)", "Czech Republic": "Czechia", "Egypt, Arab Rep.": "Egypt", "Gambia, The": "Gambia",

 "Iran, Islamic Rep.": "Iran", "Kyrgyz Republic": "Kyrgyzstan", "Korea, Rep.": "Korea, South", "Lao PDR": "Laos", "St. Lucia": "Saint Lucia",

"Korea, Dem. People’s Rep.": "Korea, North", "Russian Federation": "Russia", "South Sudan": "Sudan", "Slovak Republic": "Slovakia", 

 "Syrian Arab Republic": "Syria", "United States": "US", "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",

 "Venezuela, RB": "Venezuela"})
print("airpol data shape: {}".format(df_airpol.shape))

df_airpol = df_airpol[df_airpol["Country Name"].isin(df_train_raw["Country_Region"].values)]

print("airpol data shape: {}".format(df_airpol.shape))

df_airpol.columns.values
df_airpol = df_airpol[["Country Name", "2017"]]

df_airpol = df_airpol.rename(columns={"Country Name": "Country_Region", "2017": "PM25_2017"})

df_airpol.head()
print("skew: {}".format(np.log1p(df_airpol["PM25_2017"]).skew()))

df_airpol.loc[:, "PM25_2017"] = np.log1p(df_airpol["PM25_2017"])

df_airpol.head()
df_airpol.to_csv("pm25-global-air-pollution-2017.csv", index=False)
df_train_raw.head()
date_valid_start = df_test_raw[df_test_raw["Date"].isin(df_train_raw["Date"])]["Date"].unique().min()

date_valid_end = df_test_raw[df_test_raw["Date"].isin(df_train_raw["Date"])]["Date"].unique().max()

print("valid start: ", date_valid_start)

print("valid end: ", date_valid_end)
df_train_raw["Province_State"].fillna(df_train_raw["Country_Region"], inplace=True)
df_train_delta = df_train_raw[~df_train_raw["Date"].isin(df_train_enriched["Date"].values)]

df_train_delta.shape
base_date = pd.to_datetime("2020-01-01")

df_train_delta.loc[:, "days_since"] = (pd.to_datetime(df_train_delta["Date"]) - base_date).dt.days
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
df_train_enriched_updated = pd.concat([df_train_enriched, df_train_delta], sort=False)

df_train_enriched_updated.shape
df_train_clean = df_train_enriched_updated.drop(["Id"], axis=1)

df_test_clean = df_test_raw[~df_test_raw["Date"].isin(df_train_enriched_updated["Date"])]

df_test_clean = df_test_clean.drop(["ForecastId"], axis=1)

print("train shape: ", df_train_clean.shape)

print("test shape: ", df_test_clean.shape)
df_train_clean["ConfirmedCases"] = np.log1p(df_train_clean["ConfirmedCases"])

df_train_clean["Fatalities"] = np.log1p(df_train_clean["Fatalities"])

df_test_clean["ConfirmedCases"] = None

df_test_clean["Fatalities"] = None
df_test_clean["ConfirmedCases"] = df_test_clean["ConfirmedCases"].astype('float')

df_test_clean["Fatalities"] = df_test_clean["Fatalities"].astype('float')

df_test_clean["Fatalities"].dtype
df = pd.concat([df_train_clean, df_test_clean], sort=False).reset_index(drop=True)

print(df.shape)

print(df.columns.values)
df["Date"] = pd.to_datetime(df["Date"])
df["Province_State"].fillna(value = df["Country_Region"], inplace = True)
df = df.merge(df_airpol.drop_duplicates(subset=["Country_Region"]), how="left")

df.head()
df_env = df_env.rename(columns={"Country": "Country_Region"})

df_env.columns.values
df = df.merge(df_env.drop_duplicates(subset=["Country_Region"]), how="left")

df.columns.values
def get_trend(df, col):

  trend_col = "trend_{}".format(col)

  df[trend_col] = (df[col] - df.groupby(["Country_Region", "Province_State"])[col].shift().fillna(-999)) / (df.groupby(["Country_Region", "Province_State"])[col].shift().fillna(0) + 0.0001)

  df.loc[df[trend_col] > 100, trend_col] = 0

  

  return df



def get_lagged_value(df, col, start, end):

  for lag in list(range(start, end)):

    lagged_col = "{}-{}D".format(col, lag)

    print(lagged_col)

    df[lagged_col] = df.groupby(["Country_Region", "Province_State"])[col].shift(lag).fillna(0)



  return df



def get_trendline(y, order=1):

  x = list(range(0, len(y)))

  coeffs = np.polyfit(x, y, order)

  slope = coeffs[-2]



  return float(slope)
df_lagged = df.copy(deep=True)

df_lagged = get_trend(df_lagged, "ConfirmedCases")

# df_lagged.head()
df_lagged = get_trend(df_lagged, "Fatalities")

# df_lagged.head()
df_lagged["F2C_ratio"] = df_lagged["Fatalities"] / (df_lagged["ConfirmedCases"] + 0.0001)

df_lagged.loc[(df_lagged["ConfirmedCases"] == 0) & (df_lagged["F2C_ratio"] > 100), "F2C_ratio"] = 0

# df_lagged.head()
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
df_lagged["rolling_mean_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=3).mean())

df_lagged["rolling_std_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=3).std())

df_lagged["rolling_median_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=3).median())
df_lagged["rolling_slope_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].rolling(3).apply(get_trendline, raw=True).reset_index(drop=True)

df_lagged["rolling_slope_confirmedcases"].describe()
df_lagged["rolling_slope_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].rolling(3).apply(get_trendline, raw=True).reset_index(drop=True)

df_lagged["rolling_slope_fatalities"].describe()
df_lagged["rolling_mean_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=3).mean())

df_lagged["rolling_std_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=3).std())

df_lagged["rolling_median_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=3).median())
df_lagged = get_lagged_value(df_lagged, "rolling_mean_confirmedcases", 1, 4)

df_lagged = get_lagged_value(df_lagged, "rolling_std_confirmedcases", 1, 4)

df_lagged = get_lagged_value(df_lagged, "rolling_median_confirmedcases", 1, 4)

df_lagged = get_lagged_value(df_lagged, "rolling_slope_confirmedcases", 1, 4)

df_lagged = get_lagged_value(df_lagged, "rolling_mean_fatalities", 1, 4)

df_lagged = get_lagged_value(df_lagged, "rolling_std_fatalities", 1, 4)

df_lagged = get_lagged_value(df_lagged, "rolling_median_fatalities", 1, 4)

df_lagged = get_lagged_value(df_lagged, "rolling_slope_fatalities", 1, 4)
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
rolling_cols = ["rolling_mean_confirmedcases", "rolling_median_confirmedcases", "rolling_std_confirmedcases", "rolling_slope_confirmedcases",

                "rolling_mean_fatalities", "rolling_std_fatalities", "rolling_median_fatalities", "rolling_slope_fatalities"]



for i in range(1, 4):

  for col in rolling_cols:

    df_lagged["{}-{}D".format(col, i)].fillna(0, inplace=True)
start = df_lagged["days_since"].min()

end = df_lagged["days_since"].max()



cutoff_day = (pd.to_datetime("2020-04-01") - base_date).days

print("cutoff_day: ", cutoff_day)



for col in measures:

  print("{}: {}".format(col, (df_lagged[col] > 0).sum()))



  grouped_measures = df_lagged.loc[df_lagged[col] > 0].groupby(["Country_Region", "Province_State"]).head(1)[["Country_Region", "Province_State", "days_since"]]



  for key, row in grouped_measures.iterrows():

#     print(row["Country_Region"], row["Province_State"], row["days_since"])

    df_lagged.loc[(df_lagged["Country_Region"] == row["Country_Region"]) & (df_lagged["Province_State"] == row["Province_State"]) & (df_lagged["days_since"] > row["days_since"]), col] = 1

  

#   print("{}: {}".format(col, (df_lagged[col] > 0).sum()))



  df_lagged.loc[df_lagged["days_since"] > cutoff_day, col] = 1



#   print("after hack: {}: {}".format(col, (df_lagged[col] > 0).sum()))
for col in measures:

  df_lagged.loc[:, col] = (df_lagged.loc[:, col] > 0) * 1
for key, group in df_lagged.groupby(["Country_Region", "Province_State"]):

  length = len(group[~group["ConfirmedCases"].isna()])

  x = range(0, length)

  yc = group[~group["ConfirmedCases"].isna()]["ConfirmedCases"]

  yf = group[~group["Fatalities"].isna()]["Fatalities"]



  for order in list(range(1, 6)):

    cc_slope = get_trendline(yc, order)

    f_slope = get_trendline(yf, order)



    for i, row in group.iterrows():

      df_lagged.at[i, "cc_slope_o{}".format(order)] = cc_slope

      df_lagged.at[i, "f_slope_o{}".format(order)] = f_slope
from sklearn.metrics import mean_squared_error



def rmsle(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    return np.sqrt(mean_squared_error(y_pred, y_true))
df_train = df_lagged[df_lagged["Date"] < date_valid_start]

df_valid = df_lagged[(df_lagged["Date"] >= date_valid_start) & (df_lagged["Date"] <= date_valid_end)]

df_valid.shape
target = "ConfirmedCases"

droppable = ["ConfirmedCases", "Fatalities", "trend_ConfirmedCases", "trend_Fatalities", "F2C_ratio", 

             'rolling_mean_confirmedcases', 'rolling_std_confirmedcases', 'rolling_median_confirmedcases',

             'rolling_mean_fatalities', 'rolling_std_fatalities', 'rolling_median_fatalities',

             'rolling_slope_confirmedcases', 'rolling_slope_fatalities']

y_train = df_train[target]

X_train = df_train.drop(droppable, axis = 1)

y_valid = df_valid[target]

X_valid = df_valid.drop(droppable, axis = 1)
mse_scorer = make_scorer(rmsle, greater_is_better=True, needs_proba=False)
X_train.drop(["Date"], axis=1, inplace=True)

X_valid.drop(["Date"], axis=1, inplace=True)
model_ConfirmedCases = LGBMRegressor(bagging_fraction=0.8036720162947864, 

                                     bagging_freq=3,

                                     boosting_type='gbdt', 

                                     colsample_bytree=0.8123400218011508, 

                                     importance_type='split',

                                     learning_rate=0.005, 

                                     max_depth=-1, 

                                     min_child_samples=10,

                                     min_child_weight=0.001, 

                                     min_split_gain=0.0008337731377606208,

                                     n_estimators=500, 

                                     n_jobs=-1, 

                                     num_leaves=50, 

                                     random_seed=42, 

                                     reg_alpha=0.23492361007344623,

                                     reg_lambda=0.16520570479644592, 

                                     silent=True, 

                                     subsample_for_bin=24000,  

                                     verbose=-1)

model_ConfirmedCases.fit(X_train, y_train)

# model_ConfirmedCases.booster_.save_model(os.path.join(dirname, "model_confirmed_cases.txt"))
y_pred_confirmedCases = model_ConfirmedCases.predict(X_valid)

print("test score: {:.3f}".format(rmsle(y_valid, y_pred_confirmedCases)))
df_allTrain = pd.concat([df_train, df_valid])

y = pd.concat([y_train, y_valid])

X = pd.concat([X_train, X_valid])

model_ConfirmedCases.fit(X, y)
feature_imp = pd.DataFrame(sorted(zip(model_ConfirmedCases.feature_importances_/np.max(model_ConfirmedCases.feature_importances_),X.columns)), columns=['Value','Feature'])



plt.figure(figsize=(12, 18))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-cc.png')
target_1 = "Fatalities"

y_train = df_train[target_1]

y_valid = df_valid[target_1]
model_Fatalities = LGBMRegressor(bagging_fraction=0.9268437667105205, 

                                 bagging_freq=2,

                                 boosting_type='gbdt', 

                                 colsample_bytree=0.720747840119609, 

                                 importance_type='split',

                                 learning_rate=0.005, 

                                 max_depth=-1, 

                                 min_child_samples=5,

                                 min_child_weight=0.001, 

                                 min_split_gain=0.00020193816906540233,

                                 n_estimators=500, 

                                 n_jobs=-1, 

                                 num_leaves=25, 

                                 random_seed=42,

                                 reg_alpha=0.523260008305012,

                                 reg_lambda=0.18681303717301287, 

                                 silent=True, 

                                 subsample_for_bin=13000, 

                                 verbose=-1)



model_Fatalities.fit(X_train, y_train)

# model_Fatalities.booster_.save_model(os.path.join(dirname, "model_fatalities.txt"))
y_pred_fatalities = model_Fatalities.predict(X_valid)

print("test score: {:.3f}".format(rmsle(y_valid, y_pred_fatalities)))
y = pd.concat([y_train, y_valid], sort=False)

X = pd.concat([X_train, X_valid], sort=False)

model_Fatalities.fit(X, y)
feature_imp = pd.DataFrame(sorted(zip(model_Fatalities.feature_importances_/np.max(model_Fatalities.feature_importances_),X.columns)), columns=['Value','Feature'])



plt.figure(figsize=(12, 18))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-f.png')
date_test_end = df_test_raw["Date"].unique().max()

cutoff = (pd.to_datetime(date_valid_start) - base_date).days

start = (pd.to_datetime(date_valid_end) - base_date).days

end = (pd.to_datetime(date_test_end) - base_date).days

print(start)

print(end)

print(cutoff)
df_test = df_lagged[df_lagged["days_since"] >= cutoff].copy(deep=True)

df_test.drop([  'trend_ConfirmedCases'

              , 'trend_Fatalities'

              , 'F2C_ratio'

              , 'Date'

              , 'rolling_mean_confirmedcases'

              , 'rolling_std_confirmedcases'

              , 'rolling_median_confirmedcases'

              , 'rolling_slope_confirmedcases'

              , 'rolling_mean_fatalities'

              , 'rolling_std_fatalities'

              , 'rolling_median_fatalities'

              , 'rolling_slope_fatalities'], axis=1, inplace=True)
day = start + 1



while day <= end: 

  df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] <= start), target] = y_pred_confirmedCases

  df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] <= start), target_1] = y_pred_fatalities



  df_test = get_trend(df_test, target)

  df_test = get_trend(df_test, target_1)

  df_test["F2C_ratio"] = df_test[target_1] / (df_test[target] + 0.0001)



  df_test.loc[(df_test[target] == 0) & (df_test["F2C_ratio"] > 100), "F2C_ratio"] = 0



  df_test = get_lagged_value(df_test, target, 1, 7)

  df_test = get_lagged_value(df_test, target_1, 1, 7)



  df_test.loc[(df_test["{}-1D".format(target)] == 0) & (df_test["trend_ConfirmedCases"] > 100), "trend_ConfirmedCases"] = 0

  df_test.loc[(df_test["{}-1D".format(target)] > 0) & (df_test["trend_ConfirmedCases"] > 100), "trend_ConfirmedCases"] = 1

  df_test.loc[(df_test["{}-1D".format(target_1)] == 0) & (df_test["trend_Fatalities"] > 100), "trend_Fatalities"] = 0

  df_test.loc[(df_test["{}-1D".format(target_1)] > 0) & (df_test["trend_Fatalities"] > 100), "trend_Fatalities"] = 1



  df_test = get_lagged_value(df_test, "trend_ConfirmedCases", 1, 7)

  df_test = get_lagged_value(df_test, "trend_Fatalities", 1, 7)

  df_test = get_lagged_value(df_test, "F2C_ratio", 1, 7)



  df_test["rolling_mean_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=3).mean())

  df_test["rolling_std_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=3).std())

  df_test["rolling_median_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=3).median())



  df_test["rolling_slope_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].rolling(3).apply(get_trendline, raw=True).reset_index(drop=True)



  df_test["rolling_mean_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=3).mean())

  df_test["rolling_std_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=3).std())

  df_test["rolling_median_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=3).median())



  df_test["rolling_slope_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].rolling(3).apply(get_trendline, raw=True).reset_index(drop=True)



  df_test = get_lagged_value(df_test, "rolling_mean_confirmedcases", 1, 4)

  df_test = get_lagged_value(df_test, "rolling_std_confirmedcases", 1, 4)

  df_test = get_lagged_value(df_test, "rolling_median_confirmedcases", 1, 4)

  df_test = get_lagged_value(df_test, "rolling_slope_confirmedcases", 1, 4)

  df_test = get_lagged_value(df_test, "rolling_mean_fatalities", 1, 4)

  df_test = get_lagged_value(df_test, "rolling_std_fatalities", 1, 4)

  df_test = get_lagged_value(df_test, "rolling_median_fatalities", 1, 4)

  df_test = get_lagged_value(df_test, "rolling_slope_fatalities", 1, 4)



  for i in range(1, 4):

    for col in rolling_cols:

      df_test["{}-{}D".format(col, i)].fillna(0, inplace=True)



  X = df_test.drop(droppable, axis = 1)



  df_test.loc[:, target] = model_ConfirmedCases.predict(X)

  df_test.loc[:, target_1] = model_Fatalities.predict(X)



  day = day + 1
(df_test.isna()).sum()
df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] <= start), target] = y_pred_confirmedCases

df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] <= start), target_1] = y_pred_fatalities
df_test["ConfirmedCases"] = np.expm1(df_test["ConfirmedCases"])

df_test["Fatalities"] = np.expm1(df_test["Fatalities"])
df_test = df_test[["days_since"

                  , "Province_State"

                  , "Country_Region"

                  , "ConfirmedCases"

                  , "Fatalities"]]



df_test.head()
df_temp = df_test_raw.copy(deep=True)

df_temp["Province_State"].fillna(value = df_temp["Country_Region"], inplace=True)



df_temp["Date"] = pd.to_datetime(df_temp["Date"])

df_temp["days_since"] = (df_temp["Date"] - base_date).dt.days

df_temp = pd.merge(df_temp, df_test, how="left", on=["days_since", "Province_State", "Country_Region"])

df_temp.shape
df_temp[df_temp[target_1] < 0]
df_temp.loc[:, target] = np.ceil(df_temp[target]).astype(int)

df_temp.loc[:, target_1] = np.ceil(df_temp[target_1]).astype(int)
df_temp = df_temp[["ForecastId", target, target_1]]

df_temp.to_csv("submission.csv", index=False)
df_temp.describe()