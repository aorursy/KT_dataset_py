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
SEED = 42

np.random.seed(SEED)



from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import LabelEncoder

from catboost import Pool, CatBoostRegressor

import matplotlib.pyplot as plt

import seaborn as sns

import gc
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")



train_weather = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

test_weather = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")



metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
merged_train = pd.merge(train, metadata, how="left", on=["building_id"])

merged_test = pd.merge(test, metadata, how="left", on=["building_id"])



del train, test, metadata

gc.collect()
def simple_mem_reduce(df):

    for col in df.columns:

        if df[col].dtype == int:

            m = df[col].max()

            if m > np.iinfo(np.uint32).max:

                df[col] = df[col].astype(np.uint64)

            elif m > np.iinfo(np.uint16).max:

                df[col] = df[col].astype(np.uint32)

            elif m > np.iinfo(np.uint8).max:

                df[col] = df[col].astype(np.uint16)

            elif m < np.iinfo(np.uint8).max:

                df[col] = df[col].astype(np.uint8)

                

        elif df[col].dtype == float:

            m = df[col].max()

            if m > np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float64)

            elif m > np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float32)

            elif m < np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float16)

        

    return df
merged_train = simple_mem_reduce(merged_train)

merged_test = simple_mem_reduce(merged_test)
train_weather = simple_mem_reduce(train_weather)

test_weather = simple_mem_reduce(test_weather)
train_df = pd.merge(merged_train, train_weather, \

                    how="left", on=["site_id", "timestamp"])

test_df = pd.merge(merged_test, test_weather, \

                    how="left", on=["site_id", "timestamp"])



del merged_train, merged_test, train_weather, test_weather

gc.collect()
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])

test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])





train_df["year"] = train_df["timestamp"].dt.year.astype(np.uint16)

train_df["month"] = train_df["timestamp"].dt.month.astype(np.uint8)

train_df["day"] = train_df["timestamp"].dt.day.astype(np.uint8)

train_df["hour"] = train_df["timestamp"].dt.hour.astype(np.uint8)



test_df["year"] = test_df["timestamp"].dt.year.astype(np.uint16)

test_df["month"] = test_df["timestamp"].dt.month.astype(np.uint8)

test_df["day"] = test_df["timestamp"].dt.day.astype(np.uint8)

test_df["hour"] = test_df["timestamp"].dt.hour.astype(np.uint8)



del train_df["timestamp"], test_df["timestamp"]

gc.collect()
encoder = LabelEncoder()



train_df["primary_use"] = encoder.fit_transform(train_df["primary_use"]).astype(np.uint8)

test_df["primary_use"] = encoder.fit_transform(test_df["primary_use"]).astype(np.uint8)
gc.collect()
def RMSLE(y_true, y_pred, *args, **kwargs):

    return np.sqrt(mean_squared_log_error(y_true, y_pred))
# train_df = train_df.fillna(-999)

# test_df = test_df.fillna(-999)
y = train_df["meter_reading"]



del train_df["meter_reading"]

gc.collect()
model = CatBoostRegressor(loss_function="RMSE",

                           eval_metric="RMSE",

                           task_type="GPU",

                           learning_rate=0.01,

                           iterations=180000,

                           l2_leaf_reg=5,

                           random_seed=42,

                           od_type="Iter",

                           depth=7,

                           early_stopping_rounds=3000,

                           border_count=32

                          )



train_data = Pool(train_df[:-100000], label=np.log1p(y)[:-100000])

valid_data = Pool(train_df[-100000:], label=np.log1p(y)[-100000:])



clf = model.fit(train_data,

                    eval_set=valid_data,

                    use_best_model=True,

                    verbose=2000)
val_pred = clf.predict(train_df[-100000:])

val_pred = np.exp(val_pred)



RMSLE(y[-100000:], val_pred)
pred = clf.predict(test_df)
sub = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")
sub["meter_reading"] = np.exp(pred)

sub.to_csv("submission.csv", index=False)

sub.head(10)