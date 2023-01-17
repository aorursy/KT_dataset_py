# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import math



import pandas as pd

import numpy as np

import scipy as sp

import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer, mean_squared_error



import lightgbm as lgb

from lightgbm import LGBMRegressor



from hyperopt import hp, tpe

from hyperopt.fmin import fmin



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_env = pd.read_csv("/kaggle/input/global-environmental-factors/env.csv")

print("env: ", df_env.shape)

df_airpol = pd.read_csv("/kaggle/input/pm25-global-air-pollution/pm25-global-air-pollution-2017.csv")

print("pol: ", df_airpol.shape)

df_pop = pd.read_csv("/kaggle/input/world-population-by-country-state/country_population.csv")

print("pop: ", df_pop.shape)
df_env.isna().sum()
df_env.loc[df_env["Country_Region"].isin(['Norway', 'Finland', 'Iceland', 'Estonia']), "wind"] = 4.689151

df_env.loc[df_env["Country_Region"].isin(['Maldives']), "wind"] = np.mean([2.698925, 3.494908])

df_env.loc[df_env["Country_Region"].isin(['Bahrain']), "wind"] = np.mean([3.728877, 3.173667, 4.525724])

df_env.loc[df_env["Country_Region"].isin(['Antigua and Barbuda']), "wind"] = np.mean([3.586282, 3.378886, 2.749947])

df_env.loc[df_env["Country_Region"].isin(['Saint Vincent and the Grenadines']), "wind"] = 3.515223

df_env.loc[df_env["Country_Region"].isin(['Malta']), "wind"] = np.mean([3.078635, 2.648621])

df_env.loc[df_env["Country_Region"].isin(['Seychelles']), "wind"] = 2.736786

df_env.loc[df_env["Country_Region"].isin(['Saint Kitts and Nevis']), "wind"] = 3.515223

df_env.loc[df_env["Country_Region"].isin(['Grenada']), "wind"] = 3.515223

df_env.loc[df_env["Country_Region"].isin(['Saint Lucia']), "wind"] = 3.515223

df_env.loc[df_env["Country_Region"].isin(['Barbados']), "wind"] = 3.515223

df_env.loc[df_env["Country_Region"].isin(['Monaco']), "wind"] = np.mean([5.745106, 3.369222])



cols_na = ["accessibility_to_cities", "elevation", "aspect", "slope",

          "tree_canopy_cover", "isothermality", "rain_coldestQuart", 

          "rain_driestMonth", "rain_driestQuart", "rain_mean_annual",

          "rain_seasonailty", "rain_warmestQuart", "rain_wettestMonth",

          "rain_wettestQuart", "temp_annual_range", "temp_coldestQuart", 

           "temp_diurnal_range", "temp_driestQuart", "temp_max_warmestMonth", 

           "temp_mean_annual", "temp_min_coldestMonth", "temp_seasonality", 

           "temp_warmestQuart", "temp_wettestQuart", "cloudiness"

          ]



for c in cols_na:

    country = df_env.loc[df_env[c].isna(), "Country_Region"].unique()

    print(c, country)

    

    if 'Maldives' in country:

        df_env.loc[df_env[c].isna(), c] = df_env[df_env["Country_Region"] == "India"][c].mean()

    else:

        df_env.loc[df_env[c].isna(), c] = df_env[df_env["Country_Region"] == "Denmark"][c].mean()



# 'Norway', 'Finland', 'Iceland', 'Estonia': "Denmark" 4.689151

# 'Maldives': 'India', 'Sri lanka' np.mean(2.698925, 3.494908)

# 'Bahrain': 'Iran', 'Qatar', 'Saudi Arabia' np.mean([3.728877, 3.173667, 4.525724])

# 'Antigua and Barbuda': np.mean([3.586282, 3.378886, 2.749947])

# 'Saint Vincent and the Grenadines': 3.515223

# 'Malta': np.mean([3.078635, 2.648621])

# 'Seychelles': 2.736786

# 'Saint Kitts and Nevis': 3.515223

# 'Grenada': 3.515223

# 'Saint Lucia': 3.515223

# 'Barbados': 3.515223

# 'Monaco': np.mean([5.745106, 3.369222])
df_pop = df_pop[~df_pop["Country_Region"].isna()]

df_pop.drop(['quarantine', 'schools', 'restrictions'], axis=1, inplace=True)

df_pop.isna().sum()
df_pop.dtypes
cols_na = ["pop", "tests", "testpop", "density", "medianage",

          "urbanpop", "hospibed", "smokers", "sex0", "sex14",

          "sex25", "sex54", "sex64", "sex65plus", "sexratio",

          "lung", "femalelung", "malelung"]



for c in cols_na:

    df_pop[c] = df_pop.groupby(["Country_Region"])[c].transform(lambda x: x.fillna(x.mean()))

    

for c in cols_na:

    df_pop[c].fillna(df_pop[c].mean(), inplace=True)

# df_pop.isna().sum()
dirname = "/kaggle/input"

train_enriched_filename = "covid19-forecasting-data-with-containment-measures/train-enriched-with-containment_v5.csv"

train_filename = "covid19-global-forecasting-week-4/train.csv"

test_filename = "covid19-global-forecasting-week-4/test.csv"

df_train_enriched = pd.read_csv(os.path.join(dirname, train_enriched_filename))

df_train_raw = pd.read_csv(os.path.join(dirname, train_filename))

df_test_raw = pd.read_csv(os.path.join(dirname, test_filename))

df_train_raw.shape
date_valid_start = df_test_raw[df_test_raw["Date"].isin(df_train_raw["Date"])]["Date"].unique().min()

date_valid_end = df_test_raw[df_test_raw["Date"].isin(df_train_raw["Date"])]["Date"].unique().max()

print("valid start: ", date_valid_start)

print("valid end: ", date_valid_end)
df_train_clean = df_train_enriched.drop(["Id"], axis=1)

df_test_clean = df_test_raw[~df_test_raw["Date"].isin(df_train_enriched["Date"])]

df_test_clean = df_test_clean.drop(["ForecastId"], axis=1)

print("train shape: ", df_train_clean.shape)

print("test shape: ", df_test_clean.shape)
base_date = pd.to_datetime("2020-01-01")

base_date
df_test_clean["days_since"] = (pd.to_datetime(df_test_clean["Date"]) - base_date).dt.days

df_test_clean["days_since"].unique()
df_train_clean["ConfirmedCases"] = np.log1p(df_train_clean["ConfirmedCases"])

df_train_clean["Fatalities"] = np.log1p(df_train_clean["Fatalities"])

df_test_clean["ConfirmedCases"] = None

df_test_clean["Fatalities"] = None
print("cc skew: {}".format(df_train_clean["ConfirmedCases"].skew()))

print("ft skew: {}".format(df_train_clean["Fatalities"].skew()))
df_test_clean["ConfirmedCases"] = df_test_clean["ConfirmedCases"].astype('float')

df_test_clean["Fatalities"] = df_test_clean["Fatalities"].astype('float')

df_test_clean["Fatalities"].dtype
df = pd.concat([df_train_clean, df_test_clean], sort=False).reset_index(drop=True)

print(df.shape)

print(df.columns.values)
df["Province_State"].fillna(df["Country_Region"], inplace=True)
df = pd.merge(df, df_airpol.drop_duplicates(subset=["Country_Region"]), how="left")

df.shape
df = pd.merge(df, df_env.drop_duplicates(subset=["Country_Region"]), how="left")

df.shape
df.dtypes
df_pop.columns.values
df = pd.merge(df, df_pop.drop_duplicates(subset=["Country_Region", "Province_State"]), how="left")

df.shape
df["Date"] = pd.to_datetime(df["Date"])
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

df_lagged.shape
df_lagged = get_trend(df_lagged, "Fatalities")
df_lagged["F2C_ratio"] = df_lagged["Fatalities"] / (df_lagged["ConfirmedCases"] + 0.0001)

df_lagged.loc[(df_lagged["ConfirmedCases"] == 0) & (df_lagged["F2C_ratio"] > 100), "F2C_ratio"] = 0
s = 1

w = 3

e = w + 1
df_lagged = get_lagged_value(df_lagged, "ConfirmedCases", s, e)

df_lagged = get_lagged_value(df_lagged, "Fatalities", s, e)
df_lagged.loc[(df_lagged["ConfirmedCases-1D"] == 0) & (df_lagged["trend_ConfirmedCases"] > 100), "trend_ConfirmedCases"] = 0
df_lagged.loc[(df_lagged["Fatalities-1D"] == 0) & (df_lagged["trend_Fatalities"] > 100), "trend_Fatalities"] = 0
df_lagged = get_lagged_value(df_lagged, "trend_ConfirmedCases", s, e)

df_lagged = get_lagged_value(df_lagged, "trend_Fatalities", s, e)

df_lagged = get_lagged_value(df_lagged, "F2C_ratio", s, e)
df_lagged.loc[:,"Province_State"] = df_lagged["Province_State"].astype("category")

df_lagged.loc[:,"Country_Region"] = df_lagged["Country_Region"].astype("category")
measures = ['medicare', 'social_measure', 'resumed_measure', 'travel_measure', 'isolation_measure',

            'edu_measure', 'biz_measure', 'awareness_measure', 'lockdown_measure', 'border_measure', 

            'testing_measure', 'financial_measure', 'restaurant_measure', 'tracking_measure',

            'hygiene_measure', 'transport_measure']



for col in measures:

  df_lagged[col].fillna(value=0, inplace=True)
df_lagged["rolling_mean_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=w).mean())

df_lagged["rolling_std_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=w).std())

df_lagged["rolling_median_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=w).median())
df_lagged["rolling_slope_confirmedcases"] = df_lagged.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].rolling(w).apply(get_trendline, raw=True).reset_index(drop=True)

df_lagged["rolling_slope_confirmedcases"].describe()
df_lagged["rolling_slope_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].rolling(w).apply(get_trendline, raw=True).reset_index(drop=True)

df_lagged["rolling_slope_fatalities"].describe()
df_lagged["rolling_mean_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=w).mean())

df_lagged["rolling_std_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=w).std())

df_lagged["rolling_median_fatalities"] = df_lagged.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=w).median())
df_lagged = get_lagged_value(df_lagged, "rolling_mean_confirmedcases", s, e)

df_lagged = get_lagged_value(df_lagged, "rolling_std_confirmedcases", s, e)

df_lagged = get_lagged_value(df_lagged, "rolling_median_confirmedcases", s, e)

df_lagged = get_lagged_value(df_lagged, "rolling_slope_confirmedcases", s, e)

df_lagged = get_lagged_value(df_lagged, "rolling_mean_fatalities", s, e)

df_lagged = get_lagged_value(df_lagged, "rolling_std_fatalities", s, e)

df_lagged = get_lagged_value(df_lagged, "rolling_median_fatalities", s, e)

df_lagged = get_lagged_value(df_lagged, "rolling_slope_fatalities", s, e)
cols_float = ["ConfirmedCases", "ConfirmedCases-1D", "ConfirmedCases-2D", "ConfirmedCases-3D",

              "trend_ConfirmedCases", "trend_ConfirmedCases-1D", "trend_ConfirmedCases-2D", "trend_ConfirmedCases-3D", 

              "Fatalities", "Fatalities-1D", "Fatalities-2D", "Fatalities-3D", 

              "trend_Fatalities", "trend_Fatalities-1D", "trend_Fatalities-2D", "trend_Fatalities-3D",

              "F2C_ratio", "F2C_ratio-1D", "F2C_ratio-2D", "F2C_ratio-3D"]



for col in cols_float:

  df_lagged[col] = df_lagged[col].astype(float)
rolling_cols = ["rolling_mean_confirmedcases", "rolling_median_confirmedcases", "rolling_std_confirmedcases", "rolling_slope_confirmedcases",

                "rolling_mean_fatalities", "rolling_std_fatalities", "rolling_median_fatalities", "rolling_slope_fatalities"]



for i in range(s, e):

  for col in rolling_cols:

    df_lagged["{}-{}D".format(col, i)].fillna(0, inplace=True)
start = df_lagged["days_since"].min()

end = df_lagged["days_since"].max()



for col in measures:

  print("{}: {}".format(col, (df_lagged[col] > 0).sum()))



  grouped_measures = df_lagged.loc[df_lagged[col] > 0].groupby(["Country_Region", "Province_State"]).head(1)[["Country_Region", "Province_State", "days_since"]]



  for key, row in grouped_measures.iterrows():

    print(row["Country_Region"], row["Province_State"], row["days_since"])

    df_lagged.loc[(df_lagged["Country_Region"] == row["Country_Region"]) & (df_lagged["Province_State"] == row["Province_State"]) & (df_lagged["days_since"] > row["days_since"]), col] = 1

  

  print("{}: {}".format(col, (df_lagged[col] > 0).sum()))
(df_lagged["biz_measure"] > 0).sum()
for col in measures:

  df_lagged.loc[:, col] = (df_lagged.loc[:, col] > 0) * 1
# df_lagged_filtered = df_lagged.groupby(["Country_Region", "Province_State"], as_index=False).apply(lambda x: x[x["ConfirmedCases"] > 0])

df_nonzero = pd.DataFrame()

grouped_data = df_lagged.loc[df_lagged["ConfirmedCases"] > 0].groupby(["Country_Region", "Province_State"]).head(1)[["Country_Region", "Province_State", "days_since"]]



for key, row in grouped_data.iterrows():

#     print(row["Country_Region"], row["Province_State"], row["days_since"])

    df_nonzero = df_nonzero.append({"Country_Region": row["Country_Region"], 

                                   "Province_State": row["Province_State"], 

                                   "days_since": row["days_since"]}, ignore_index=True)

    

df_nonzero.head(2)
for key, group in df_lagged.groupby(["Country_Region", "Province_State"]):

  

  day_min = df_nonzero.loc[(df_nonzero["Country_Region"] == key[0]) & (df_nonzero["Province_State"] == key[1]), "days_since"]

  day_min = day_min.iloc[0]  

#   print(key[0], key[1], day_min)



  yc = group[(~group["ConfirmedCases"].isna()) & (group["days_since"] >= day_min)]["ConfirmedCases"]

  yf = group[(~group["Fatalities"].isna()) & (group["days_since"] >= day_min)]["Fatalities"]

  length = len(yc)

  x = range(0, length)

  

  for order in list(range(1, 6)):

    cc_slope = get_trendline(yc, order)

    f_slope = get_trendline(yf, order)



    for i, row in group.iterrows():

      df_lagged.at[i, "cc_slope_o{}".format(order)] = cc_slope

      df_lagged.at[i, "f_slope_o{}".format(order)] = f_slope
def rmsle(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    return np.sqrt(mean_squared_error(y_true, y_pred))



mse_scorer = make_scorer(rmsle, greater_is_better=True, needs_proba=False)
df_lagged["malelung"].describe()

# df_lagged["malelung"] = df_lagged.groupby(["Country_Region"]).transform(lambda x: x.fillna(x.mean()))

# df_lagged["malelung"].fillna(df_lagged["malelung"].mean(), inplace=True)
cols_na = ["PM25_2017", "accessibility_to_cities", "elevation", "aspect", "slope",

          "tree_canopy_cover", "isothermality", "rain_coldestQuart", 

          "rain_driestMonth", "rain_driestQuart", "rain_mean_annual",

          "rain_seasonailty", "rain_warmestQuart", "rain_wettestMonth",

          "rain_wettestQuart", "temp_annual_range", "temp_coldestQuart", 

           "temp_diurnal_range", "temp_driestQuart", "temp_max_warmestMonth", 

           "temp_mean_annual", "temp_min_coldestMonth", "temp_seasonality", 

           "temp_warmestQuart", "temp_wettestQuart", "cloudiness", 

           "pop", "tests", "testpop", "density", "medianage",

          "urbanpop", "hospibed", "smokers", "sex0", "sex14",

          "sex25", "sex54", "sex64", "sex65plus", "sexratio",

          "lung", "femalelung", "malelung"]



for c in cols_na:

#     print(c, df_lagged[c].dtype)

    count = df_lagged[c].isna().sum()

    

    if count > 0:

        print(c, count)

        df_lagged[c] = df_lagged.groupby(["Country_Region"])[c].transform(lambda x: x.fillna(x.mean()))

        df_lagged[c].fillna(df_lagged[c].mean(), inplace=True)

        

    print(c, df_lagged[c].dtype)
df_lagged[["pop", "tests", "testpop", "density", "medianage",

          "urbanpop", "smokers", "sex0", "lung", "femalelung", "malelung"]].describe()
pop_cols = ["pop", "tests", "testpop", "density", "medianage",

            "urbanpop", "smokers", "lung", "femalelung", "malelung"]



for c in pop_cols:

    df_lagged[c] = np.log1p(df_lagged[c])
df_train = df_lagged[df_lagged["Date"] < date_valid_start]

df_valid = df_lagged[(df_lagged["Date"] >= date_valid_start) & (df_lagged["Date"] <= date_valid_end)]

df_valid.shape
df_train_new = pd.DataFrame()



for key, group in df_train.groupby(["Country_Region", "Province_State"]):

  

  day_min = df_nonzero.loc[(df_nonzero["Country_Region"] == key[0]) & (df_nonzero["Province_State"] == key[1]), "days_since"]

  day_min = day_min.iloc[0] 

  

  df_train_new = df_train_new.append(group[group["days_since"] >= day_min])



print(df_train.shape, df_train_new.shape)
target = "ConfirmedCases"

droppable = ["ConfirmedCases", "Fatalities", "trend_ConfirmedCases", "trend_Fatalities", "F2C_ratio", 

             'rolling_mean_confirmedcases', 'rolling_std_confirmedcases', 'rolling_median_confirmedcases',

             'rolling_mean_fatalities', 'rolling_std_fatalities', 'rolling_median_fatalities',

             'rolling_slope_confirmedcases', 'rolling_slope_fatalities']

  

droppable_c = ["border_measure", "transport_measure", "financial_measure", "resumed_measure",

               "tracking_measure", "restaurant_measure", "hygiene_measure", "biz_measure",

               "awareness_measure", "medicare", "travel_measure", "testing_measure", "rolling_median_fatalities-1D",

               "rolling_median_fatalities-2D", "rolling_median_fatalities-3D", "rolling_mean_fatalities-2D",

               "rolling_mean_fatalities-3D", "Fatalities-3D", "trend_Fatalities-2D", "trend_Fatalities-3D"]



droppable_f = ["border_measure", "transport_measure", "tracking_measure", "restaurant_measure",

              "financial_measure", "resumed_measure", "hygiene_measure", "biz_measure", 

               "isolation_measure", "medicare", "awareness_measure", "edu_measure", "social_measure",

               "travel_measure", "rain_mean_annual", "rain_wettestQuart", "temp_mean_annual", "temp_coldestQuart",

              "rain_driestQuart", "tree_canopy_cover", "temp_warmestQuart", "temp_seasonality", "medianage", "testpop", "sex0"]



y_train = df_train_new[target]

X_train = df_train_new.drop(droppable, axis = 1)

y_valid = df_valid[target]

X_valid = df_valid.drop(droppable, axis = 1)



X_train.drop(['Date'], axis=1, inplace=True)

X_valid.drop(['Date'], axis=1, inplace=True)



X_train_c = X_train.drop(droppable_c, axis=1)

X_valid_c = X_valid.drop(droppable_c, axis=1)
X_valid_c.head()
len(X_train_c.columns.values)
len(X_valid_c.columns.values)
df_lagged[cols_na].dtypes
n_splits = 10
model_ConfirmedCases = lgb.LGBMRegressor( boosting_type="goss",

                                          # n_estimators=850, test score: 0.13193831

                                          # learning_rate=0.01, test score: 0.13193831

                                          n_estimators=850,

                                          learning_rate=0.01,

                                          max_bin=255,

                                          top_rate=0.9458886954629803,

                                          other_rate=0.036843813509725376,

                                          num_leaves=55,

                                          colsample_bytree=0.9362293586249408,

                                          subsample_for_bin=7000,

                                          min_child_samples=30, 

                                          min_split_gain=0.009628782573584817,

                                          reg_alpha=0.012588241918997418, 

                                          reg_lambda=0.2685018640839245,

                                          seed=int(2**n_splits),

                                          bagging_seed=int(2**n_splits),

                                          drop_seed=int(2**n_splits),

                                          verbose=-1,

                                          n_jobs=-1

                                      )



model_ConfirmedCases.fit(X_train_c, y_train)
y_pred_confirmedCases = model_ConfirmedCases.predict(X_valid_c)

print("test score: {:.8f}".format(rmsle(y_valid, y_pred_confirmedCases)))
y = pd.concat([y_train, y_valid])

X = pd.concat([X_train_c, X_valid_c])

model_ConfirmedCases.fit(X, y)
feature_imp = pd.DataFrame(sorted(zip(model_ConfirmedCases.feature_importances_/np.max(model_ConfirmedCases.feature_importances_),X.columns)), columns=['Value','Feature'])



plt.figure(figsize=(12, 18))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-cc.png')
target_1 = "Fatalities"

y_train = df_train_new[target_1]

y_valid = df_valid[target_1]

X_train_f = X_train.drop(droppable_f, axis=1)

X_valid_f = X_valid.drop(droppable_f, axis=1)
model_Fatalities = lgb.LGBMRegressor( n_estimators=1600,

                                      learning_rate=0.005,

                                      max_bin=255,

                                      top_rate=0.7952928990812882,

                                      other_rate=0.02348027656640178,

                                      num_leaves=25,

                                      colsample_bytree=0.8886565168523001,

                                      subsample_for_bin=4000,

                                      min_child_samples=10, 

                                      min_split_gain=0.009810258634362159,

                                      reg_alpha=0.0022726097664520946, 

                                      reg_lambda=0.4864860341055296,

                                      seed=int(2**n_splits),

                                      bagging_seed=int(2**n_splits),

                                      drop_seed=int(2**n_splits),

                                      verbose=-1,

                                      n_jobs=-1

#                                       random_seed=42

                                      )



model_Fatalities.fit(X_train_f, y_train)
y_pred_fatalities = model_Fatalities.predict(X_valid_f)

print("test score: {:.8f}".format(rmsle(y_valid, y_pred_fatalities)))
y = pd.concat([y_train, y_valid], sort=False)

X = pd.concat([X_train_f, X_valid_f], sort=False)

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
df_test.shape
df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] <= start), target] = y_pred_confirmedCases

df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] <= start), target_1] = y_pred_fatalities
day = start + 1



while day <= end: 

  df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] <= start), target] = y_pred_confirmedCases

  df_test.loc[(df_test["days_since"] >= cutoff) & (df_test["days_since"] <= start), target_1] = y_pred_fatalities



  df_test = get_trend(df_test, target)

  df_test = get_trend(df_test, target_1)

  df_test["F2C_ratio"] = df_test[target_1] / (df_test[target] + 0.0001)



  df_test.loc[(df_test[target] == 0) & (df_test["F2C_ratio"] > 100), "F2C_ratio"] = 0



  df_test = get_lagged_value(df_test, target, s, e)

  df_test = get_lagged_value(df_test, target_1, s, e)



  df_test.loc[(df_test["{}-1D".format(target)] == 0) & (df_test["trend_ConfirmedCases"] > 100), "trend_ConfirmedCases"] = 0

  df_test.loc[(df_test["{}-1D".format(target)] > 0) & (df_test["trend_ConfirmedCases"] > 100), "trend_ConfirmedCases"] = 1

  df_test.loc[(df_test["{}-1D".format(target_1)] == 0) & (df_test["trend_Fatalities"] > 100), "trend_Fatalities"] = 0

  df_test.loc[(df_test["{}-1D".format(target_1)] > 0) & (df_test["trend_Fatalities"] > 100), "trend_Fatalities"] = 1



  df_test = get_lagged_value(df_test, "trend_ConfirmedCases", s, e)

  df_test = get_lagged_value(df_test, "trend_Fatalities", s, e)

  df_test = get_lagged_value(df_test, "F2C_ratio", s, e)



  df_test["rolling_mean_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=w).mean())

  df_test["rolling_std_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=w).std())

  df_test["rolling_median_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].apply(lambda x: x.rolling(center=False, window=w).median())



  df_test["rolling_slope_confirmedcases"] = df_test.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].rolling(w).apply(get_trendline, raw=True).reset_index(drop=True)



  df_test["rolling_mean_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=w).mean())

  df_test["rolling_std_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=w).std())

  df_test["rolling_median_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].apply(lambda x: x.rolling(center=False, window=w).median())



  df_test["rolling_slope_fatalities"] = df_test.groupby(["Country_Region", "Province_State"])["Fatalities"].rolling(w).apply(get_trendline, raw=True).reset_index(drop=True)



  df_test = get_lagged_value(df_test, "rolling_mean_confirmedcases", s, e)

  df_test = get_lagged_value(df_test, "rolling_std_confirmedcases", s, e)

  df_test = get_lagged_value(df_test, "rolling_median_confirmedcases", s, e)

  df_test = get_lagged_value(df_test, "rolling_slope_confirmedcases", s, e)

  df_test = get_lagged_value(df_test, "rolling_mean_fatalities", s, e)

  df_test = get_lagged_value(df_test, "rolling_std_fatalities", s, e)

  df_test = get_lagged_value(df_test, "rolling_median_fatalities", s, e)

  df_test = get_lagged_value(df_test, "rolling_slope_fatalities", s, e)



  for i in range(s, e):

    for col in rolling_cols:

      df_test["{}-{}D".format(col, i)].fillna(0, inplace=True)



  X = df_test.drop(droppable, axis = 1)

  X_c = X.drop(droppable_c, axis=1)

  X_f = X.drop(droppable_f, axis=1)



  df_test.loc[:, target] = model_ConfirmedCases.predict(X_c)

  df_test.loc[:, target_1] = model_Fatalities.predict(X_f)



  day = day + 1
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
df_test.shape
df_temp.isna().sum()
df_temp.describe()
df_temp.loc[df_temp[target_1] < 0, target_1] = 0

# df_temp[df_temp[target_1] < 0].describe()
df_temp.loc[:, target] = np.ceil(df_temp[target]).astype(int)

df_temp.loc[:, target_1] = np.ceil(df_temp[target_1]).astype(int)
df_temp = df_temp[["ForecastId", target, target_1]]

df_temp.to_csv("submission.csv", index=False)
df_temp.describe()