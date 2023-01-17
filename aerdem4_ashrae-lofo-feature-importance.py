# install the latest LOFO

!pip install lofo-importance==0.2.4
import numpy as np

import pandas as pd



train_df = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

train_df.shape
# sample 1M from 20M to calculate the importance faster



train_df = train_df.sample(10**6, random_state=0)

train_df.head()
building_df = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")



train_df = train_df.merge(building_df, on="building_id", how="left")

train_df.head()
weather_df = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")



train_df = train_df.merge(weather_df, on=["site_id", "timestamp"], how="left")

train_df.head()
train_df["day_of_week"] = pd.to_datetime(train_df["timestamp"]).dt.weekday

train_df["hour"] = pd.to_datetime(train_df["timestamp"]).dt.hour

train_df.head()
# model from https://www.kaggle.com/corochann/ashrae-simple-lgbm-submission



import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder



le_pu = LabelEncoder()

train_df["primary_use_le"] = le_pu.fit_transform(train_df["primary_use"])



cat_cols = ['building_id', 'meter', 'site_id', 'primary_use_le', "day_of_week"]



params = {'num_leaves': 31,

          'objective': 'regression',

          'learning_rate': 0.1,

          "boosting": "gbdt",

          "bagging_freq": 5,

          "bagging_fraction": 0.1,

          "feature_fraction": 0.9,

          "num_rounds": 600

          }



model = lgb.LGBMRegressor(**params)
from sklearn.model_selection import KFold

from lofo import LOFOImportance, Dataset, plot_importance



# time based cv split

train_df.sort_values("timestamp", inplace=True)

cv = KFold(n_splits=4, shuffle=False, random_state=0)



train_df["target"] = np.log1p(train_df["meter_reading"])

features = ['square_feet', 'year_built', 'floor_count', 'air_temperature', 'cloud_coverage', 'dew_temperature',

               'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'hour'] + cat_cols



dataset = Dataset(df=train_df, target="target", features=features)



lofo_imp = LOFOImportance(dataset, model=model, cv=cv, scoring="neg_mean_squared_error", fit_params={"categorical_feature": cat_cols})



importance_df = lofo_imp.get_importance()



plot_importance(importance_df, figsize=(12, 12))