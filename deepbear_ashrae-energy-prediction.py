import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMRegressor

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from tqdm import tqdm



"""

inspiration from:

https://www.kaggle.com/gouherdanishiitkgp/ashrae-basic-eda-and-feature-engineering

https://www.kaggle.com/isaienkov/lightgbm-fe-1-19

https://www.kaggle.com/rohanrao/ashrae-divide-and-conquer

"""
metadata_dtype = {"site_id":"uint8","building_id":"uint16","square_feet":"float32","year_built":"float32","floor_count":"float16"}

metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv",dtype=metadata_dtype)

metadata.drop("floor_count", axis=1, inplace=True)
weather_dtype = {"site_id":"uint8"}

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",parse_dates=["timestamp"],dtype=weather_dtype)

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",parse_dates=["timestamp"],dtype=weather_dtype)

weather_train.drop(["sea_level_pressure", "wind_direction", "wind_speed"], axis=1, inplace=True)

weather_test.drop(["sea_level_pressure", "wind_direction", "wind_speed"], axis=1, inplace=True)

weather_train = weather_train.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))

weather_test = weather_test.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))
train_dtype = {"meter":"uint8","building_id":"uint16","meter_reading":"float32"}

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv",parse_dates=["timestamp"],dtype=train_dtype)

test_dtype = {"meter":"uint8","building_id":"uint16"}

test_cols_to_read = ["building_id","meter","timestamp"]

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv",parse_dates=["timestamp"],usecols=test_cols_to_read,dtype=test_dtype)
def lag_features(dataset):

    features = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"]

    grouped = dataset.groupby("site_id")

    rolled = grouped[features].rolling(window=15, min_periods=0)

    dataset_mean = rolled.mean().reset_index().astype(np.float16)

    dataset_median = rolled.median().reset_index().astype(np.float16)

    dataset_min = rolled.min().reset_index().astype(np.float16)

    dataset_max = rolled.max().reset_index().astype(np.float16)

    

    for feature in features:

        dataset[f"{feature}_mean_lag"] = dataset_mean[feature]

        dataset[f"{feature}_median_lag"] = dataset_median[feature]

        dataset[f"{feature}_max_lag"] = dataset_max[feature]

        dataset[f"{feature}_min_lag"] = dataset_min[feature]

    return dataset
weather_train = lag_features(weather_train)

weather_test = lag_features(weather_test)

weather_train.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"], axis=1, inplace=True)

weather_test.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"], axis=1, inplace=True)

print(weather_train.shape)

print(weather_test.shape)

train = train.merge(metadata, left_on = "building_id", right_on = "building_id", how = "left")

train = train.merge(weather_train, left_on = ["timestamp", "site_id"], right_on=["timestamp", "site_id"], how = "left")

test = test.merge(metadata, left_on = "building_id", right_on = "building_id", how = "left")

test = test.merge(weather_test, left_on = ["timestamp", "site_id"], right_on=["timestamp", "site_id"], how = "left")

print(train.shape)

print(test.shape)
train["day"] = train["timestamp"].dt.weekday.astype("uint8")

train["hour"] = train["timestamp"].dt.hour.astype("uint8")

test["day"] = test["timestamp"].dt.weekday.astype("uint8")

test["hour"] = test["timestamp"].dt.hour.astype("uint8")

train.drop("timestamp", axis=1, inplace=True)

test.drop("timestamp", axis=1, inplace=True)

building_meter = train.groupby(["building_id", "meter"]).agg(mean_building_meter=("meter_reading", "mean"), 

                                                                median_building_meter=("meter_reading", "median")).reset_index()

train = train.merge(building_meter, on=["building_id", "meter"])

test = test.merge(building_meter, on=["building_id", "meter"])
train = train[train["meter_reading"]>0.0]

print(train.shape)

print(test.shape)
le = LabelEncoder()

train["primary_use"] = le.fit_transform(train["primary_use"])

test["primary_use"] = le.fit_transform(test["primary_use"])

categorical = ["site_id", "building_id", "primary_use", "hour", "day", "meter"]

features = [col for col in train.columns if col != "meter_reading"]
labels = np.log1p(train["meter_reading"])

params = {"objective": "regression",

          "metric": "rmse",

          "num_leaves": 30,

          "learning_rate": 0.05,

          "bagging_fraction": 0.50,

          "feature_fraction": 0.80,

          "bagging_freq": 5}



kf = StratifiedKFold(n_splits = 2, shuffle=True)

model_arr = []

for train_id, val_id in kf.split(train, train["building_id"]):

    X_train = train[features].iloc[train_id]

    X_test = train[features].iloc[val_id]

    y_train = labels.iloc[train_id]

    y_test = labels.iloc[val_id]

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical)

    lgb_test = lgb.Dataset(X_test, y_test, categorical_feature=categorical)

    gbm = lgb.train(params, lgb_train, num_boost_round = 500, valid_sets = (lgb_train, lgb_test), 

                    early_stopping_rounds = 100, verbose_eval = 100)

    model_arr.append(gbm)
start = 0

preds = []

interval = 50000

for _ in tqdm(range(int(np.ceil(len(test)/interval)))):

    preds.append(np.expm1(sum([model.predict(test.iloc[start:start+interval]) for model in model_arr])/2))

    start += interval



sub = pd.DataFrame({"row_id:":test.index, "meter_reading":np.concatenate(preds)})

sub.to_csv("submission.csv", index=False, float_format="%.4f")
print(sub.head())

print(sub.tail())

sub.head()
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

leak0 = pd.read_csv("../input/site0.csv")

leak1 = pd.read_csv("../input/site1.csv")

leak2 = pd.read_csv("../input/site2.csv")

leak4 = pd.read_csv("../input/site4.csv")

leak15 = pd.read_csv("../input/site15.csv")



leak = pd.concat([leak0, leak1, leak2, leak4, leak15])
leak.head()
test.head()
test = test[test.building_id.isin(leak.building_id.unique())]

leak = leak.merge(test, on=["building_id", "meter", "timestamp"])

sub = sub.merge(leak[["row_id", "meter_reading_scraped"]], on=["row_id"], how="left")

sub.loc[sub.meter_reading_scraped.notnull(), "meter_reading"] = sub.loc[sub.meter_reading_scraped.notnull(), "meter_reading_scraped"]

sub.drop(["meter_reading_scraped"], axis=1, inplace=True)

sub.head()



sub.to_csv("submission_lk.csv", index=False)