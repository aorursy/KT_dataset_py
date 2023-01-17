import pandas as pd 

import numpy as np



import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly

plotly.offline.init_notebook_mode()



import matplotlib.pyplot as plt

import matplotlib.animation as animation

from IPython.display import HTML

%matplotlib inline



from tqdm import tqdm



def RMSLE(pred,actual):

    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))
import lightgbm as lgb

import numpy as np

import pandas as pd



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder



import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

from scipy.optimize import curve_fit



%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.metrics import log_loss

from sklearn.preprocessing import OneHotEncoder



import xgboost as xgb



from tensorflow.keras.optimizers import Nadam

from sklearn.metrics import mean_squared_error

import tensorflow as tf

import tensorflow.keras.layers as KL

from datetime import timedelta

import numpy as np

import pandas as pd





import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge



import datetime

import gc

from tqdm import tqdm
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")



region_metadata = pd.read_csv("/kaggle/input/covid19-forecasting-metadata/region_metadata.csv")

region_date_metadata = pd.read_csv("/kaggle/input/covid19-forecasting-metadata/region_date_metadata.csv")
train = train.merge(test[["ForecastId", "Province_State", "Country_Region", "Date"]], on = ["Province_State", "Country_Region", "Date"], how = "left")

display(train.head())

test = test[~test.Date.isin(train.Date.unique())]

display(test.head())



df = pd.concat([train, test], sort = False)

df.head()
df["geo"] = df.Country_Region.astype(str) + ": " + df.Province_State.astype(str)

df.loc[df.Province_State.isna(), "geo"] = df[df.Province_State.isna()].Country_Region



df.ConfirmedCases = df.groupby("geo")["ConfirmedCases"].cummax()

df.Fatalities = df.groupby("geo")["Fatalities"].cummax()



df = df.merge(region_metadata, on = ["Country_Region", "Province_State"])

df = df.merge(region_date_metadata, on = ["Country_Region", "Province_State", "Date"], how = "left")

df.continent = LabelEncoder().fit_transform(df.continent)

df.Date = pd.to_datetime(df.Date, format = "%Y-%m-%d")

df.sort_values(["geo", "Date"], inplace = True)

df.head()
DAYS_SINCE_CASES = [1, 10, 50, 100, 500, 1000, 5000, 10000]

min_date_train = np.min(df[~df.Id.isna()].Date)

max_date_train = np.max(df[~df.Id.isna()].Date)



min_date_test = np.min(df[~df.ForecastId.isna()].Date)

max_date_test = np.max(df[~df.ForecastId.isna()].Date)



n_dates_test = len(df[~df.ForecastId.isna()].Date.unique())



print("Train date range:", str(min_date_train), " - ", str(max_date_train))

print("Test date range:", str(min_date_test), " - ", str(max_date_test))



# creating lag features

for lag in range(1, 41):

    df[f"lag_{lag}_cc"] = df.groupby("geo")["ConfirmedCases"].shift(lag)

    df[f"lag_{lag}_ft"] = df.groupby("geo")["Fatalities"].shift(lag)

    df[f"lag_{lag}_rc"] = df.groupby("geo")["Recoveries"].shift(lag)



for case in DAYS_SINCE_CASES:

    df = df.merge(df[df.ConfirmedCases >= case].groupby("geo")["Date"].min().reset_index().rename(columns = {"Date": f"case_{case}_date"}), on = "geo", how = "left")
def prepare_features(df, gap):

    

    df["perc_1_ac"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap}_ft"] - df[f"lag_{gap}_rc"]) / df[f"lag_{gap}_cc"]

    df["perc_1_cc"] = df[f"lag_{gap}_cc"] / df.population

    

    df["diff_1_cc"] = df[f"lag_{gap}_cc"] - df[f"lag_{gap + 1}_cc"]

    df["diff_2_cc"] = df[f"lag_{gap + 1}_cc"] - df[f"lag_{gap + 2}_cc"]

    df["diff_3_cc"] = df[f"lag_{gap + 2}_cc"] - df[f"lag_{gap + 3}_cc"]

    

    df["diff_1_ft"] = df[f"lag_{gap}_ft"] - df[f"lag_{gap + 1}_ft"]

    df["diff_2_ft"] = df[f"lag_{gap + 1}_ft"] - df[f"lag_{gap + 2}_ft"]

    df["diff_3_ft"] = df[f"lag_{gap + 2}_ft"] - df[f"lag_{gap + 3}_ft"]

    

    df["diff_123_cc"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap + 3}_cc"]) / 3

    df["diff_123_ft"] = (df[f"lag_{gap}_ft"] - df[f"lag_{gap + 3}_ft"]) / 3



    df["diff_change_1_cc"] = df.diff_1_cc / df.diff_2_cc

    df["diff_change_2_cc"] = df.diff_2_cc / df.diff_3_cc

    

    df["diff_change_1_ft"] = df.diff_1_ft / df.diff_2_ft

    df["diff_change_2_ft"] = df.diff_2_ft / df.diff_3_ft



    df["diff_change_12_cc"] = (df.diff_change_1_cc + df.diff_change_2_cc) / 2

    df["diff_change_12_ft"] = (df.diff_change_1_ft + df.diff_change_2_ft) / 2

    

    df["change_1_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 1}_cc"]

    df["change_2_cc"] = df[f"lag_{gap + 1}_cc"] / df[f"lag_{gap + 2}_cc"]

    df["change_3_cc"] = df[f"lag_{gap + 2}_cc"] / df[f"lag_{gap + 3}_cc"]



    df["change_1_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 1}_ft"]

    df["change_2_ft"] = df[f"lag_{gap + 1}_ft"] / df[f"lag_{gap + 2}_ft"]

    df["change_3_ft"] = df[f"lag_{gap + 2}_ft"] / df[f"lag_{gap + 3}_ft"]



    df["change_123_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 3}_cc"]

    df["change_123_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 3}_ft"]

    

    for case in DAYS_SINCE_CASES:

        df[f"days_since_{case}_case"] = (df[f"case_{case}_date"] - df.Date).astype("timedelta64[D]")

        df.loc[df[f"days_since_{case}_case"] < gap, f"days_since_{case}_case"] = np.nan



    df["country_flag"] = df.Province_State.isna().astype(int)

    df["density"] = df.population / df.area

    

    # target variable is log of change from last known value

    df["target_cc"] = np.log1p(df.ConfirmedCases) - np.log1p(df[f"lag_{gap}_cc"])

    df["target_ft"] = np.log1p(df.Fatalities) - np.log1p(df[f"lag_{gap}_ft"])

    

    features = [

        f"lag_{gap}_cc",

        f"lag_{gap}_ft",

        f"lag_{gap}_rc",

        "perc_1_ac",

        "perc_1_cc",

        "diff_1_cc",

        "diff_2_cc",

        "diff_3_cc",

        "diff_1_ft",

        "diff_2_ft",

        "diff_3_ft",

        "diff_123_cc",

        "diff_123_ft",

        "diff_change_1_cc",

        "diff_change_2_cc",

        "diff_change_1_ft",

        "diff_change_2_ft",

        "diff_change_12_cc",

        "diff_change_12_ft",

        "change_1_cc",

        "change_2_cc",

        "change_3_cc",

        "change_1_ft",

        "change_2_ft",

        "change_3_ft",

        "change_123_cc",

        "change_123_ft",

        "days_since_1_case",

        "days_since_10_case",

        "days_since_50_case",

        "days_since_100_case",

        "days_since_500_case",

        "days_since_1000_case",

        "days_since_5000_case",

        "days_since_10000_case",

        "country_flag",

        "lat",

        "lon",

        "continent",

        "population",

        "area",

        "density",

        "target_cc",

        "target_ft"

    ]

    

    return df[features]
def build_predict_lgbm(df_train, df_test, gap):

    

    df_train.dropna(subset = ["target_cc", "target_ft", f"lag_{gap}_cc", f"lag_{gap}_ft"], inplace = True)

    

    target_cc = df_train.target_cc

    target_ft = df_train.target_ft

    

    test_lag_cc = df_test[f"lag_{gap}_cc"].values

    test_lag_ft = df_test[f"lag_{gap}_ft"].values

    

    df_train.drop(["target_cc", "target_ft"], axis = 1, inplace = True)

    df_test.drop(["target_cc", "target_ft"], axis = 1, inplace = True)

    

    categorical_features = ["continent"]

    

    dtrain_cc = lgb.Dataset(df_train, label = target_cc, categorical_feature = categorical_features)

    dtrain_ft = lgb.Dataset(df_train, label = target_ft, categorical_feature = categorical_features)



    model_cc = lgb.train(LGB_PARAMS, train_set = dtrain_cc, num_boost_round = 200)

    model_ft = lgb.train(LGB_PARAMS, train_set = dtrain_ft, num_boost_round = 200)

    

    # inverse transform from log of change from last known value

    y_pred_cc = np.expm1(model_cc.predict(df_test, num_boost_round = 200) + np.log1p(test_lag_cc))

    y_pred_ft = np.expm1(model_ft.predict(df_test, num_boost_round = 200) + np.log1p(test_lag_ft))

    

    return y_pred_cc, y_pred_ft, model_cc, model_ft
def predict_mad(df_test, gap, val = False):

    

    df_test["avg_diff_cc"] = (df_test[f"lag_{gap}_cc"] - df_test[f"lag_{gap + 3}_cc"]) / 3

    df_test["avg_diff_ft"] = (df_test[f"lag_{gap}_ft"] - df_test[f"lag_{gap + 3}_ft"]) / 3



    if val:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (1 - MAD_FACTOR) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / VAL_DAYS

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (1 - MAD_FACTOR) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / VAL_DAYS

    else:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (1 - MAD_FACTOR) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / n_dates_test

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (1 - MAD_FACTOR) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / n_dates_test



    return y_pred_cc, y_pred_ft
SEED = 24



LGB_PARAMS = {"objective": "regression",

              "num_leaves": 5,

              "learning_rate": 0.013,

              "bagging_fraction": 0.91,

              "feature_fraction": 0.81,

              "reg_alpha": 0.13,

              "reg_lambda": 0.13,

              "metric": "rmse",

              "seed": SEED

             }

VAL_DAYS = 7

MAD_FACTOR = 0.5
df_train = df[~df.Id.isna()]

df_test_full = df[~df.ForecastId.isna()]
df_preds_val = []

df_preds_test = []



for date in df_test_full.Date.unique():

    

    print("[INFO] Date:", date)

    

    # ignore date already present in train data

    if date in df_train.Date.values:

        df_pred_test = df_test_full.loc[df_test_full.Date == date, ["ForecastId", "ConfirmedCases", "Fatalities"]].rename(columns = {"ConfirmedCases": "ConfirmedCases_test", "Fatalities": "Fatalities_test"})

    else:

        df_test = df_test_full[df_test_full.Date == date]

        

        gap = (pd.Timestamp(date) - max_date_train).days

        

        if gap <= VAL_DAYS:

            val_date = max_date_train - pd.Timedelta(VAL_DAYS, "D") + pd.Timedelta(gap, "D")



            df_build = df_train[df_train.Date < val_date]

            df_val = df_train[df_train.Date == val_date]

            

            X_build = prepare_features(df_build, gap)

            X_val = prepare_features(df_val, gap)

            

            y_val_cc_lgb, y_val_ft_lgb, _, _ = build_predict_lgbm(X_build, X_val, gap)

            y_val_cc_mad, y_val_ft_mad = predict_mad(df_val, gap, val = True)

            

            df_pred_val = pd.DataFrame({"Id": df_val.Id.values,

                                        "ConfirmedCases_val_lgb": y_val_cc_lgb,

                                        "Fatalities_val_lgb": y_val_ft_lgb,

                                        "ConfirmedCases_val_mad": y_val_cc_mad,

                                        "Fatalities_val_mad": y_val_ft_mad,

                                       })



            df_preds_val.append(df_pred_val)



        X_train = prepare_features(df_train, gap)

        X_test = prepare_features(df_test, gap)



        y_test_cc_lgb, y_test_ft_lgb, model_cc, model_ft = build_predict_lgbm(X_train, X_test, gap)

        y_test_cc_mad, y_test_ft_mad = predict_mad(df_test, gap)

        

        if gap == 1:

            model_1_cc = model_cc

            model_1_ft = model_ft

            features_1 = X_train.columns.values

        elif gap == 14:

            model_14_cc = model_cc

            model_14_ft = model_ft

            features_14 = X_train.columns.values

        elif gap == 28:

            model_28_cc = model_cc

            model_28_ft = model_ft

            features_28 = X_train.columns.values



        df_pred_test = pd.DataFrame({"ForecastId": df_test.ForecastId.values,

                                     "ConfirmedCases_test_lgb": y_test_cc_lgb,

                                     "Fatalities_test_lgb": y_test_ft_lgb,

                                     "ConfirmedCases_test_mad": y_test_cc_mad,

                                     "Fatalities_test_mad": y_test_ft_mad,

                                    })

    

    df_preds_test.append(df_pred_test)
df = df.merge(pd.concat(df_preds_val, sort = False), on = "Id", how = "left")

df = df.merge(pd.concat(df_preds_test, sort = False), on = "ForecastId", how = "left")



rmsle_cc_lgb = np.sqrt(mean_squared_error(np.log1p(df[~df.ConfirmedCases_val_lgb.isna()].ConfirmedCases), np.log1p(df[~df.ConfirmedCases_val_lgb.isna()].ConfirmedCases_val_lgb)))

rmsle_ft_lgb = np.sqrt(mean_squared_error(np.log1p(df[~df.Fatalities_val_lgb.isna()].Fatalities), np.log1p(df[~df.Fatalities_val_lgb.isna()].Fatalities_val_lgb)))



rmsle_cc_mad = np.sqrt(mean_squared_error(np.log1p(df[~df.ConfirmedCases_val_mad.isna()].ConfirmedCases), np.log1p(df[~df.ConfirmedCases_val_mad.isna()].ConfirmedCases_val_mad)))

rmsle_ft_mad = np.sqrt(mean_squared_error(np.log1p(df[~df.Fatalities_val_mad.isna()].Fatalities), np.log1p(df[~df.Fatalities_val_mad.isna()].Fatalities_val_mad)))



print("LGB CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_lgb, 2))

print("LGB FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_lgb, 2))

print("LGB Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_lgb + rmsle_ft_lgb) / 2, 2))

print("MAD CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_mad, 2))

print("MAD FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_mad, 2))

print("MAD Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_mad + rmsle_ft_mad) / 2, 2))
test = df.loc[~df.ForecastId.isna(), ["ForecastId", "Country_Region", "Province_State", "Date",

                                                     "ConfirmedCases_test", "ConfirmedCases_test_lgb", "ConfirmedCases_test_mad",

                                                     "Fatalities_test", "Fatalities_test_lgb", "Fatalities_test_mad"]].reset_index()



test["ConfirmedCases"] = 0.3 * test.ConfirmedCases_test_lgb + 0.7 * test.ConfirmedCases_test_mad

test["Fatalities"] = 0.25 * test.Fatalities_test_lgb + 0.75 * test.Fatalities_test_mad



# Since LGB models don't predict these countries well

test.loc[test.Country_Region.isin(["China", "US", "Diamond Princess"]), "ConfirmedCases"] = test[test.Country_Region.isin(["China", "US", "Diamond Princess"])].ConfirmedCases_test_mad.values

test.loc[test.Country_Region.isin(["China", "US", "Diamond Princess"]), "Fatalities"] = test[test.Country_Region.isin(["China", "US", "Diamond Princess"])].Fatalities_test_mad.values



test.loc[test.Date.isin(df_train.Date.values), "ConfirmedCases"] = test[test.Date.isin(df_train.Date.values)].ConfirmedCases_test.values

test.loc[test.Date.isin(df_train.Date.values), "Fatalities"] = test[test.Date.isin(df_train.Date.values)].Fatalities_test.values



sub0 = test[["ForecastId", "ConfirmedCases", "Fatalities"]]

sub0.ForecastId = sub0.ForecastId.astype(int)



sub0.head()
train, test = train_ts, test_ts

train.shape, test.shape
display(train.head())

display(test.head())
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

train['Province_State'].fillna('', inplace=True)

test['Province_State'].fillna('', inplace=True)

train['Date'] =  pd.to_datetime(train['Date'])

test['Date'] =  pd.to_datetime(test['Date'])

train = train.sort_values(['Country_Region','Province_State','Date'])

test = test.sort_values(['Country_Region','Province_State','Date'])
FirstDate = train.groupby('Country_Region').min()['Date'].unique()[0]

train['Last Confirm'] = train['ConfirmedCases'].shift(1)

while train[(train['Last Confirm'] > train['ConfirmedCases']) & (train['Date'] > FirstDate)].shape[0] > 0:

    train['Last Confirm'] = train['ConfirmedCases'].shift(1)

    train['Last Fatalities'] = train['Fatalities'].shift(1)

    train.loc[(train['Last Confirm'] > train['ConfirmedCases']) & (train['Date'] > FirstDate),'ConfirmedCases'] = train.loc[(train['Last Confirm'] > train['ConfirmedCases']) & (train['Date'] > FirstDate),'Last Confirm']

    train.loc[(train['Last Fatalities'] > train['Fatalities']) & (train['Date'] > FirstDate),'Fatalities'] = train.loc[(train['Last Fatalities'] > train['Fatalities']) & (train['Date'] > FirstDate),'Last Fatalities']

    train['Last Confirm'] = train['ConfirmedCases'].shift(1)

    train['Last Fatalities'] = train['Fatalities'].shift(1)
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARIMA
feature_day = [1,20,50,100,200,500,1000]

def CreateInput(data):

    feature = []

    for day in feature_day:

        #Get information in train data

        data.loc[:,'Number day from ' + str(day) + ' case'] = 0

        if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):

            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        

        else:

            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       

        for i in range(0, len(data)):

            if (data['Date'].iloc[i] > fromday):

                day_denta = data['Date'].iloc[i] - fromday

                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 

        feature = feature + ['Number day from ' + str(day) + ' case']

    

    return data[feature]

pred_data_all = pd.DataFrame()

with tqdm(total=len(train['Country_Region'].unique())) as pbar:

    for country in train['Country_Region'].unique():

    #for country in ['Vietnam']:

        for province in train[(train['Country_Region'] == country)]['Province_State'].unique():

            with warnings.catch_warnings():

                warnings.filterwarnings("ignore")

                df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]

                df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                X_train = CreateInput(df_train)

                y_train_confirmed = df_train['ConfirmedCases'].ravel()

                y_train_fatalities = df_train['Fatalities'].ravel()

                X_pred = CreateInput(df_test)



                feature_use = X_pred.columns[0]

                for i in range(X_pred.shape[1] - 1,0,-1):

                    if (X_pred.iloc[0,i] > 0):

                        feature_use = X_pred.columns[i]

                        break

                idx = X_train[X_train[feature_use] == 0].shape[0] 



                adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)

                adjusted_y_train_confirmed = y_train_confirmed[idx:]

                adjusted_y_train_fatalities = y_train_fatalities[idx:] 

                idx = X_pred[X_pred[feature_use] == 0].shape[0]    

                adjusted_X_pred = X_pred[idx:][feature_use].values.reshape(-1, 1)



                pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()

                min_test_date = pred_data['Date'].min()

                model = SARIMAX(adjusted_y_train_confirmed, order=(1,1,0), 

                                measurement_error=True).fit(disp=False)

                y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

                y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values

                y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)

                model = SARIMAX(adjusted_y_train_fatalities, order=(1,1,0), 

                                measurement_error=True).fit(disp=False)

                y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

                y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values

                y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)

                pred_data['ConfirmedCases_hat'] =  y_hat_confirmed

                pred_data['Fatalities_hat'] = y_hat_fatalities

                pred_data_all = pred_data_all.append(pred_data)

        pbar.update(1)

df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')

df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0

df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0

df_val_3 = df_val.copy()
sub1 = df_val_3

submission = sub1[['ForecastId','ConfirmedCases_hat','Fatalities_hat']]

submission.columns = ['ForecastId','ConfirmedCases','Fatalities']
TARGETS = ["ConfirmedCases", "Fatalities"]

sub_df = sub0.copy()

for t in TARGETS:

    sub_df[t] = np.expm1(np.log1p(submission[t].values)*0.4 + np.log1p(sub0[t].values)*0.6)



sub_df.to_csv("submission.csv", index=False)
sub_df.head()
# sub0.head()
sub0.isna().sum()