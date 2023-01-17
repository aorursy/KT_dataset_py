# from IPython.display import HTML

# HTML('''<script>
# code_show=true; 
# function code_toggle() {
#  if (code_show){
#  $('div.input').hide();
#  } else {
#  $('div.input').show();
#  }
#  code_show = !code_show
# } 
# $( document ).ready(code_toggle);
# </script>
# The raw code for this IPython notebook is by default hidden for easier reading.
# To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

# Modeling
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

Debug = False

# Read data
NROWS = 6000000
if Debug is True: NROWS = 5000
train = pd.read_csv('../input/train.csv', nrows = NROWS, index_col = "key")
train = train.dropna()
test_df = pd.read_csv('../input/test.csv', index_col = "key")
testdex = test_df.index
def prepare_distance_features(df):
    # Distance is expected to have an impact on the fare
    df['longitude_distance'] = abs(df['pickup_longitude'] - df['dropoff_longitude'])
    df['latitude_distance'] = abs(df['pickup_latitude'] - df['dropoff_latitude'])

    # Straight distance
    df['distance_travelled'] = (df['longitude_distance'] ** 2 + df['latitude_distance'] ** 2) ** .5
    df['distance_travelled_sin'] = np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5)
    df['distance_travelled_cos'] = np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5)
    df['distance_travelled_sin_sqrd'] = np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2
    df['distance_travelled_cos_sqrd'] = np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2

    # Haversine formula for distance
    # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    R = 6371e3 # Metres
    phi1 = np.radians(df['pickup_latitude'])
    phi2 = np.radians(df['dropoff_latitude'])
    phi_chg = np.radians(df['pickup_latitude'] - df['dropoff_latitude'])
    delta_chg = np.radians(df['pickup_longitude'] - df['dropoff_longitude'])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
    d = R * c
    df['haversine'] = d

    # Bearing
    # Formula:	θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )
    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df['bearing'] = np.arctan2(y, x)

    return df

def prepare_time_features(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.replace(" UTC", "")
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    df['hour_of_day'] = df.pickup_datetime.dt.hour
    df['week'] = df.pickup_datetime.dt.week
    df['month'] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
    df['day_of_year'] = df.pickup_datetime.dt.dayofyear
    df['week_of_year'] = df.pickup_datetime.dt.weekofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["quarter"] = df.pickup_datetime.dt.quarter
    df["day_of_month"] = df.pickup_datetime.dt.day
    
    return df

# Airport Features - By Albert van Breenmen
# https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):  
    distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    
    return distance

def airport_feats(train,test_df):
    for data in [train,test_df]:
        nyc = (-74.0063889, 40.7141667)
        jfk = (-73.7822222222, 40.6441666667)
        ewr = (-74.175, 40.69)
        lgr = (-73.87, 40.77)
        data['distance_to_center'] = dist(nyc[1], nyc[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
        data['pickup_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                             data['pickup_latitude'], data['pickup_longitude'])
        data['dropoff_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                               data['dropoff_latitude'], data['dropoff_longitude'])
        data['pickup_distance_to_ewr'] = dist(ewr[1], ewr[0], 
                                              data['pickup_latitude'], data['pickup_longitude'])
        data['dropoff_distance_to_ewr'] = dist(ewr[1], ewr[0],
                                               data['dropoff_latitude'], data['dropoff_longitude'])
        data['pickup_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                              data['pickup_latitude'], data['pickup_longitude'])
        data['dropoff_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                               data['dropoff_latitude'], data['dropoff_longitude'])
    return train, test_df

# Percentile
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

# Build ime Aggregate Features
def time_agg(train, test_df, vars_to_agg, vars_be_agg):
    for var in vars_to_agg:
        agg = train.groupby(var)[vars_be_agg].agg(["sum","mean","std","skew",percentile(80),percentile(20)])
        if isinstance(var, list):
            agg.columns = pd.Index(["fare_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
        else:
            agg.columns = pd.Index(["fare_by_" + var + "_" + str(e) for e in agg.columns.tolist()]) 
        train = pd.merge(train,agg, on=var, how= "left")
        test_df = pd.merge(test_df,agg, on=var, how= "left")
    
    return train, test_df

# Clean dataset from https://www.kaggle.com/gunbl4d3/xgboost-ing-taxi-fares
def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45)]
print("Cleaning Functions Defined..")
print("Percent of Training Set with Zero and Below Fair: ", round(((train.loc[train["fare_amount"] <= 0, "fare_amount"].shape[0]/train.shape[0]) * 100),5))
print("Percent of Training Set 200 and Above Fair: ", round((train.loc[train["fare_amount"] >= 200, "fare_amount"].shape[0]/train.shape[0]) * 100,5))
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] <= 200),:]
print("\nPercent of Training Set with Zero and Below Passenger Count: ", round((train.loc[train["passenger_count"] <= 0, "passenger_count"].shape[0]/train.shape[0]) * 100,5))
print("Percent of Training Set with Nine and Above Passenger Count: ", round((train.loc[train["passenger_count"] >= 9, "passenger_count"].shape[0]/train.shape[0]) * 100,5))
train = train.loc[(train["passenger_count"] > 0) & (train["passenger_count"] <= 9),:]

# Clean Training Set
train = clean_df(train)

# Distance Features
train = prepare_distance_features(train)
test_df = prepare_distance_features(test_df)
train,test_df = airport_feats(train,test_df)

# Time Features
train = prepare_time_features(train)
test_df = prepare_time_features(test_df)

# Ratios
train["fare_to_dist_ratio"] = train["fare_amount"] / ( train["distance_travelled"]+0.0001)
train["fare_npassenger_to_dist_ratio"] = (train["fare_amount"] / train["passenger_count"]) /( train["distance_travelled"]+0.0001)

# Time Aggregate Features
train, test_df = time_agg(train, test_df,
                          vars_to_agg  = ["passenger_count", "weekday", "quarter", "month", "year", "hour_of_day",
                                          ["weekday", "month", "year"], ["hour_of_day", "weekday", "month", "year"]],
                          vars_be_agg = "fare_amount")
train_time_start = train.pickup_datetime.min()
train_time_end = train.pickup_datetime.max()
print("Train Time Starts: {}, Ends {}".format(train_time_start,train_time_end))
test_time_start = test_df.pickup_datetime.min()
test_time_end = test_df.pickup_datetime.max()
print("Test Time Starts: {}, Ends {}".format(test_time_start,test_time_end))
f, ax = plt.subplots(1,2,figsize = [10,5])
sns.countplot(train["passenger_count"], ax=ax[0])
sns.countplot(test_df["passenger_count"], ax=ax[1])
ax[0].set_title("Train Set - Passenger Count")
ax[1].set_title("Test Set - Passenger Count")
plt.show()
f, ax = plt.subplots(figsize=[6,5])
sns.kdeplot(train["fare_amount"], ax=ax)
ax.set_title("Fare Distribution")
plt.show()
def time_slicer(df, timeframes, value, color="purple"):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [12,12])
    for i,x in enumerate(timeframes):
        df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))
        ax[i].set_xlabel("")
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)
time_slicer(df=train, timeframes=['year',"day_of_year", "month", "day_of_month", "weekday", "hour_of_day"], value = "fare_amount", color="blue")
time_slicer(df=train, timeframes=['year',"day_of_year", "month", "day_of_month", "weekday", "hour_of_day"], value = "distance_travelled", color = "green")
time_slicer(df=train, timeframes=['year',"day_of_year", "month", "day_of_month", "weekday", "hour_of_day"], value = "fare_to_dist_ratio", color = "red")
time_slicer(df=train, timeframes=['year',"day_of_year", "month", "day_of_month", "weekday", "hour_of_day"], value = "fare_npassenger_to_dist_ratio", color = "orange")
# Keep Relevant Variables..
y = train.fare_amount.copy()
test_df.drop("pickup_datetime", axis = 1, inplace=True)
train = train[test_df.columns]
print("Does Train feature equal test feature?: ", all(train.columns == test_df.columns))
trainshape = train.shape
testshape = test_df.shape

# print("\nTrain DF..")
# train = reduce_mem_usage(train)
# print("\nTest DF..")
# test_df = reduce_mem_usage(test_df)

# LGBM Dataset Formating
dtrain = lgb.Dataset(train, label=y, free_raw_data=False)
print("Light Gradient Boosting Regressor: ")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse'
                }

folds = KFold(n_splits=5, shuffle=True, random_state=1)
fold_preds = np.zeros(testshape[0])
oof_preds = np.zeros(trainshape[0])
dtrain.construct()

# Fit 5 Folds
modelstart = time.time()
for trn_idx, val_idx in folds.split(train):
    clf = lgb.train(
        params=lgbm_params,
        train_set=dtrain.subset(trn_idx),
        valid_sets=dtrain.subset(val_idx),
        num_boost_round=3500, 
        early_stopping_rounds=125,
        verbose_eval=500
    )
    oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
    fold_preds += clf.predict(test_df) / folds.n_splits
    print(mean_squared_error(y.iloc[val_idx], oof_preds[val_idx]) ** .5)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
lgsub = pd.DataFrame(fold_preds,columns=["fare_amount"],index=testdex)
lgsub.to_csv("lgsub.csv",index=True,header=True)

print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
lgsub.head()
## Big shoutout to those who build many of my features.