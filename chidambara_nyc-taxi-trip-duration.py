import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
DATA_PATH = "../input"

TRAIN_FILE_PATH = os.path.join(DATA_PATH, "train.csv")

TEST_FILE_PATH = os.path.join(DATA_PATH, "test.csv")

SAMPLE_SUB_FILE_PATH = os.path.join(DATA_PATH, "sample_submission.csv")

SUB_FILE_PATH = os.path.join(DATA_PATH, "submission.csv")
train = pd.read_csv(TRAIN_FILE_PATH, index_col="id", parse_dates=["pickup_datetime", "dropoff_datetime"])
train.head()
train.info()
train.describe()
from sklearn.model_selection import train_test_split



train_set, val_set = train_test_split(train, test_size=0.2, random_state=42)

print(len(train_set), len(val_set))

train = train_set
plt.subplots(figsize=(15, 6))

train.boxplot()
from sklearn.base import BaseEstimator, TransformerMixin



class ClipOutliers(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_name, lower_limit=None, upper_limit=None):

        self.attribute_name = attribute_name

        self.lower_limit = lower_limit

        self.upper_limit = upper_limit

        

    def fit(self, df, y=None):

        return self

    

    def transform(self, df, y=None):

        if self.attribute_name not in df.columns:

            return df

        elif self.lower_limit != None:

            return df[df[self.attribute_name] > self.lower_limit]

        elif self.upper_limit != None:

            return df[df[self.attribute_name] < self.upper_limit]

        else:

            return df

        

clip_outliers_in_trip_duration = ClipOutliers("trip_duration", lower_limit=None, upper_limit=6000)

train_trip_duration_clipped = clip_outliers_in_trip_duration.fit_transform(train)

clip_outliers_in_passenger_count = ClipOutliers("passenger_count", lower_limit=0, upper_limit=None)

train_clipped = clip_outliers_in_trip_duration.fit_transform(train_trip_duration_clipped)

plt.subplots(figsize=(15, 6))

train_clipped.boxplot()
train_clipped[["passenger_count", "vendor_id"]].hist(bins=50, figsize=(15, 4))
taxi = train
taxi.plot(kind='scatter', x="pickup_longitude", y="pickup_latitude", alpha=0.4,

            s=taxi["trip_duration"]/600, label="trip_duration", figsize=(10,7),

            c=taxi["trip_duration"], cmap=plt.get_cmap("jet"), colorbar=True,)

plt.legend()
# Visualizing the pickup and dropoff locations

fig, axes = plt.subplots(1,2, sharex=True, sharey=True, figsize=(15,4))



axes[0].scatter(x=taxi["pickup_longitude"], y=taxi["pickup_latitude"], alpha=0.3,c=taxi["trip_duration"],

                cmap=plt.get_cmap("jet"),)

axes[0].legend()

axes[1].scatter(x=taxi["dropoff_longitude"], y=taxi["dropoff_latitude"], alpha=0.3,color="red")

axes[0].set_title("pickup location")

axes[0].set_xlabel("longitude")

axes[0].set_ylabel("latitude")

axes[1].set_xlabel("longitude")

axes[1].set_title("dropoff location")

plt.legend()

plt.show()
pickup_mean_latitude = taxi[(taxi["pickup_latitude"] > 40.0) & (taxi["pickup_latitude"] < 42.0)]["pickup_latitude"].mean()

pickup_mean_longitude = taxi[(taxi["pickup_longitude"] > -74.2) & (taxi["pickup_longitude"] < -73.5)]["pickup_longitude"].mean()

print("pickup_mean_latitude = {}".format(pickup_mean_latitude))

print("pickup_mean_longitude = {}".format(pickup_mean_longitude))



dropoff_mean_latitude = taxi[(taxi["dropoff_latitude"] > 40.0) & (taxi["dropoff_latitude"] < 42.0)]["pickup_latitude"].mean()

dropoff_mean_longitude = taxi[(taxi["dropoff_longitude"] > -74.2) & (taxi["dropoff_longitude"] < -73.5)]["pickup_longitude"].mean()

print("dropoff_mean_latitude = {}".format(dropoff_mean_latitude))

print("dropoff_mean_longitude = {}".format(dropoff_mean_longitude))



# Histogram of pickup and dropoff locations (latitudes and longitudes)

fig, axes = plt.subplots(2,2, sharex=False, sharey=False, figsize=(15,10))

pickup_longitude_spread = taxi[(taxi["pickup_longitude"] > -74.2) & (taxi["pickup_longitude"] < -73.5)]["pickup_longitude"]

axes[0,0].hist(pickup_longitude_spread, bins=50, color="blue")

axes[0,0].set_title("pickup longitude distribution")



pickup_latitude_spread = (taxi[(taxi["pickup_latitude"] > 40.0) & (taxi["pickup_latitude"] < 42.0)]["pickup_latitude"])

axes[0,1].hist(pickup_latitude_spread, bins=50, color="blue")

axes[0,1].set_title("pickup latitude distribution")



dropoff_latitude_spread = (taxi[(taxi["dropoff_latitude"] > 40.0) & (taxi["dropoff_latitude"] < 42.0)]["dropoff_latitude"])

axes[1,0].hist(dropoff_latitude_spread, bins=50, color="red")

axes[1,0].set_title("dropoff latitude distribution")



dropoff_longitude_spread = taxi[(taxi["dropoff_longitude"] > -74.2) & (taxi["dropoff_longitude"] < -73.5)]["dropoff_longitude"]

axes[1,1].hist(dropoff_longitude_spread, bins=50, color="red")

axes[1,1].set_title("dropoff longitude distribution")

plt.show()
# Building correlation matrix

def get_corr_info(df, feature="trip_duration"):

    corr_matrix = df.corr()

    return corr_matrix["trip_duration"].sort_values(ascending=False)
import datetime as dt



class DateAttribsAdder(BaseEstimator, TransformerMixin):

    def __init__(self, date_field):

        self.date_field = date_field

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X["month"] = X.loc[:,self.date_field].dt.month

        X["week"] = X.loc[:,self.date_field].dt.week

        X["weekday"] = X.loc[:,self.date_field].dt.weekday

        X["day"] = X.loc[:,self.date_field].dt.day

        X["hour"] = X.loc[:,self.date_field].dt.hour

        X["minute"] = X.loc[:,self.date_field].dt.minute

        X["minute_of_the_day"] = X["hour"] * 60 + X.loc[:,self.date_field].dt.minute 

        return X.drop([self.date_field, "dropoff_datetime", "minute"], axis=1, errors="ignore")



date_attribs_adder = DateAttribsAdder("pickup_datetime")

date_attribs_added = date_attribs_adder.transform(taxi)

print(date_attribs_added.head())

cat_attribs = ["vendor_id", "store_and_fwd_flag"]

taxi_cat = taxi[cat_attribs]



from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

taxi_1hot = cat_encoder.fit_transform(taxi_cat)

print(taxi_1hot.toarray())
%%time

from math import sin, cos, sqrt, atan2, radians



class DistanceAttribsAdder(BaseEstimator, TransformerMixin):

    def __init__(self, attrib_name = "distance"):

        self.attrib_name = attrib_name

        

    def fit(self, df):

        return self

    

    def transform(self, df, y=None):

        R=6373.0 # approximate radius of earth in km

        pickup_lat=np.radians(df['pickup_latitude'].values)

        pickup_lon=np.radians(df['pickup_longitude'].values)

        dropoff_lat=np.radians(df['dropoff_latitude'].values)

        dropoff_lon=np.radians(df['dropoff_longitude'].values)

        dlon = dropoff_lon - pickup_lon

        dlat = dropoff_lat - pickup_lat

        a = np.sin(dlat / 2)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2)**2

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        df["distance"] = R * c

        df.boxplot(column="distance", return_type="axes");

        plt.show()

        return df

        

distance_attribs_adder = DistanceAttribsAdder()

taxi_distance = distance_attribs_adder.transform(taxi)

print(taxi_distance.head())



# As we see outliers in the distance attribute, let's clip that out

clip_outliers_in_distance = ClipOutliers("distance", lower_limit=None, upper_limit=400)

taxi = clip_outliers_in_distance.fit_transform(taxi_distance)
## Creating a Pandas DropFeature transformer which drops the attributes from DataFrame and converts to numpy array

class DropFeature(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X.drop([self.attribute_names],axis=1,errors="ignore").values
## Creating a Pandas FeatureSelector transformer which selects the attributes from DataFrame and converts to numpy array



class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X[self.attribute_names].values
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        columns = X.columns

        selected_features = [feature for feature in self.attribute_names if feature in columns ]

        return X[selected_features]
class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_name):

        self.attribute_name = attribute_name

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X[self.attribute_name] = np.log(X[self.attribute_name].values)

        return X
# categorical pipeline

from sklearn.pipeline import Pipeline



cat_pipeline = Pipeline([

    ('selector_cat', FeatureSelector(cat_attribs)),

    ("cat_trans", OneHotEncoder()),

])

num_attribs = ['pickup_datetime','passenger_count',

 'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude',

 'trip_duration']



# Outlier rows removal piepline

outlier_pipeline = Pipeline([    

    ("clip_trip_duration", ClipOutliers("trip_duration", lower_limit=None, upper_limit=6000)),

    ("clip_passenger_count", ClipOutliers("passenger_count", lower_limit=0, upper_limit=None)),

    ("clip_pickup_longitude", ClipOutliers("pickup_longitude", lower_limit=-90, upper_limit=None)),

    ("clip_pickup_latitude", ClipOutliers("pickup_latitude", lower_limit=None, upper_limit=50)),

    ("dist_trans", DistanceAttribsAdder()),

    ("clip_distance", ClipOutliers("distance", lower_limit=None, upper_limit=400)),

])



# numerical attributes pipeline

num_pipeline = Pipeline([

    ('selector_num', DataFrameSelector(num_attribs)),

    ("date_tranform", DateAttribsAdder("pickup_datetime")),

    ("feature_dropper", DropFeature("trip_duration"))

])
# label creation pipeline

label_pipeline = Pipeline([

    ("clip_trip_duration", ClipOutliers("trip_duration", lower_limit=None, upper_limit=6000)),

    ('trip_log_transformer', LogTransformer("trip_duration")),

    ("feature_selector", FeatureSelector("trip_duration"))

])
# Create a Featureunion to join the two pipelines (numerical and categorical pipelines)



from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline),

])
# Feeding the outlier clipped transformed data into full_pipeline

outliers_clipped = outlier_pipeline.fit_transform(train)

taxi_prepared = full_pipeline.fit_transform(outliers_clipped)



# Running the label_pipeline

taxi_labels = label_pipeline.fit_transform(outliers_clipped)



print(taxi_prepared.shape)

taxi_prepared
# Validation data transformation

# Feeding the outlier clipped transformed data into full_pipeline

outliers_clipped_val = outlier_pipeline.fit_transform(val_set)

taxi_prepared_val = full_pipeline.fit_transform(outliers_clipped_val)



# Running the label_pipeline

taxi_labels_val = label_pipeline.fit_transform(outliers_clipped_val)
# Trying RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



forest_reg = RandomForestRegressor(n_estimators=50,min_samples_leaf=10, min_samples_split=15, max_features='auto', max_depth=50, bootstrap=True, n_jobs=-1)

forest_reg.fit(taxi_prepared, taxi_labels)



predictions = forest_reg.predict(taxi_prepared_val)



print(predictions[:5])

print(taxi_labels_val[:5])



score = np.sqrt(mean_squared_error(taxi_labels_val, predictions))

print("validation RMSE: {}".format(score))



# scores = cross_val_score(forest_reg, taxi_prepared, taxi_labels, scoring='neg_mean_squared_error', cv=5, test_split=0.2)

# forest_rmse_scores = np.sqrt(-scores)

# print("forest_rmse_scores: ", forest_rmse_scores)

# print("mean forest_rmse_scores: ", forest_rmse_scores.mean())

# print("std forest_rmse_scores: ", forest_rmse_scores.std())
test = pd.read_csv(TEST_FILE_PATH, index_col="id", parse_dates=["pickup_datetime"])
test.head()
outliers_clipped_test = outlier_pipeline.fit_transform(test)

test_prepared = full_pipeline.fit_transform(test)

print(test_prepared.shape)
pred_test = np.exp(forest_reg.predict(test_prepared))
submit = pd.read_csv(SAMPLE_SUB_FILE_PATH)

submit.head()
submit["trip_duration"] = pred_test

submit.head()
submit.to_csv("submission.csv", index=False)
print(forest_reg.feature_importances_)