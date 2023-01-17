#Importing the required libraries and setting up the figure parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import lightgbm as lgb
from sklearn.model_selection import KFold
from matplotlib import rcParams
dark_colors = ["#99D699", "#B2B2B2",
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]
rcParams['figure.figsize'] = (12, 9)
rcParams['figure.dpi'] = 150
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = "white"
rcParams['axes.titlesize'] = 20      
rcParams['axes.labelsize'] = 17.5
rcParams['xtick.labelsize'] = 15 
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 17.5
rcParams['patch.edgecolor'] = 'none'
rcParams['grid.color']="white"   
rcParams['grid.linestyle']="-" 
rcParams['grid.linewidth'] = 1
rcParams['grid.alpha']=1
rcParams['text.color'] = "444444"
rcParams['axes.labelcolor'] = "444444"
rcParams['ytick.color'] = "444444"
rcParams['xtick.color'] = "444444"
trips_df = pd.read_csv('../input/trip.csv')
stations_df = pd.read_csv('../input/station.csv')
stations_df.head()
stations_df.dtypes
heatmap = folium.Map([stations_df["lat"].median(),stations_df["long"].median()],zoom_start=10,tiles='Stamen Toner')
stations_df["lat"] = stations_df["lat"].apply(lambda x:str(x))
stations_df["long"] = stations_df["long"].apply(lambda x:str(x))
stations_df.head()
from folium.plugins import HeatMap
stations_loc = [[float(stations_df.lat.values[i]),float(stations_df.long.values[i])] for i in range(len(stations_df))]
heatmap.add_child(HeatMap(stations_loc,radius=10))
for index,row in stations_df.iterrows():
    folium.Marker([float(row['lat']),float(row['long'])],popup=row['name']).add_to(heatmap)
heatmap
trips_df.head()
trips_df['start_date'] = pd.to_datetime(trips_df['start_date'])
trips_df['end_date'] = pd.to_datetime(trips_df['end_date'])
start_station_info = stations_df[["id","lat","long"]]
start_station_info.columns = ["start_station_id","start_lat","start_long"]
end_station_info = stations_df[["id","lat","long"]]
end_station_info.columns = ["end_station_id","end_lat","end_long"]
trips_df = trips_df.merge(start_station_info,on="start_station_id")
trips_df = trips_df.merge(end_station_info,on="end_station_id")
trips_df.head()
plot_dict = dict()
for index,row in trips_df.iterrows():
    start_lat = row['start_lat']
    start_long = row['start_long']
    end_lat = row['end_lat']
    end_long = row['end_long']
    key = str(start_lat)+'_'+str(start_long)+'_'+str(end_lat)+'_'+str(end_long)
    if key in plot_dict:
        plot_dict[key] += 1
    else:
        plot_dict[key] = 1
start_lat = []
start_long = []
end_lat = []
end_long = []
nb_trips = []
for key,value in plot_dict.items():
    start_lat.append(float(key.split('_')[0]))
    start_long.append(float(key.split('_')[1]))
    end_lat.append(float(key.split('_')[2]))
    end_long.append(float(key.split('_')[3]))
    nb_trips.append(int(value))
temp_df = pd.DataFrame({"start_lat":start_lat,"start_long":start_long,"end_lat":end_lat,"end_long":end_long,"nb_trips":nb_trips})
temp_df.dtypes
temp_df.nb_trips.plot()
temp_df.info()
temp_df.head()
ave_lat = (temp_df.start_lat.median()+temp_df.end_lat.median())/2
ave_lon = (temp_df.start_long.median()+temp_df.end_long.median())/2
directions_map = folium.Map(location=[ave_lat, ave_lon], zoom_start=15)
for index,row in temp_df.iterrows():
    points = []
    points.append(tuple([row['start_lat'],row['start_long']]))
    points.append(tuple([row['end_lat'],row['end_long']]))
    folium.PolyLine(points,color='red',weight=row['nb_trips']/1000).add_to(directions_map)
for index,row in stations_df.iterrows():
    folium.Marker([float(row['lat']),float(row['long'])],popup=row['name']).add_to(directions_map)
directions_map
fig, ax1 = plt.subplots(figsize = (10,7))
ax1.grid(zorder=1)
ax1.xaxis.grid(False)
trip_dur = trips_df['duration'].values/60
plt.hist(trip_dur, bins = range(0,45,2),normed=True,zorder=0,color=dark_colors[1])
plt.xlabel('Trip Duration (Minutes)')
plt.ylabel('Percent of Trips')
plt.title('Trip Duration Distribution')
plt.figure(figsize=(15,12))
hist, bin_edges = np.histogram(trip_dur, range(0,45,1), normed=True)
cum_trip_dur = np.cumsum(hist)
ax2 = ax1.twinx()
ax2.plot(range(1,45,1),cum_trip_dur,c=dark_colors[0])
ax2.set_ylabel('Cumulative Proportion of Trips')
ax2.grid(b=False)
trips_df.head()
trips_df['week']=trips_df.start_date.dt.dayofweek
trips_df['start_hour'] = trips_df.start_date.dt.hour
trips_df['start_day'] = trips_df.start_date.dt.day
trips_df['end_hour'] = trips_df.end_date.dt.hour
trips_df['end_day'] = trips_df.end_date.dt.day
plt.figure(figsize=(15,12))
weekdaytrips_df = trips_df.loc[(trips_df.duration <= 7200) & (trips_df.week <5)]
weekdaytrips_df.boxplot(column="duration",by="start_hour",figsize=(15,12))
plt.ylim(0,3600)
plt.ylabel('Trip Duration (Seconds)')
plt.xlabel('Hour of Day')
plt.title('Trip Duration Distribution Over Time of Day (Week Days)')
plt.figure(figsize=(15,12))
weekendtrips_df = trips_df.loc[(trips_df.duration <= 7200) & (trips_df.week >4)]
weekendtrips_df.boxplot(column="duration",by="start_hour",figsize=(15,12))
plt.ylim(0,3600)
plt.ylabel('Trip Duration (Seconds)')
plt.xlabel('Hour of Day')
plt.title('Trip Duration Distribution Over Time of Day (Weekend days)')
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
status_df = reduce_mem_usage(pd.read_csv('../input/status.csv'))
status_df.head()
status_df.info()
status_df.time = pd.to_datetime(status_df.time)
status_df = status_df[status_df.time.dt.minute%5 ==0]
stations_df.rename(columns={"id":"station_id"},inplace=True)
stations_df.installation_date = pd.to_datetime(stations_df.installation_date)
status_df = status_df.merge(stations_df,on="station_id",how="left")
status_df.head()
status_df.reset_index(inplace=True)
status_df.drop(columns=["index"],inplace=True)
status_df["date"] = status_df.time.dt.date
status_df.head()
weather_df = reduce_mem_usage(pd.read_csv('../input/weather.csv'))
weather_df.date = pd.to_datetime(weather_df.date)
weather_df.head()
zipcode_city_dict = dict()
zipcode_city_dict[95113] = 'San Jose'
zipcode_city_dict[94301] = 'Palo Alto'
zipcode_city_dict[94107] = 'San Francisco'
zipcode_city_dict[94063] = 'Redwood City'
zipcode_city_dict[94041] = 'Mountain View'
weather_df["city"] = weather_df.zip_code.apply(lambda x:zipcode_city_dict[x])
weather_df.head()
status_df.date = pd.to_datetime(status_df.date)
status_df = status_df.merge(weather_df,how="left",on=["date","city"])
status_df.head()
status_df.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
status_df["events"] = le.fit_transform(status_df["events"])
status_df["precipitation_inches"] = le.fit_transform(status_df["precipitation_inches"])
status_df["name"] = le.fit_transform(status_df["name"])

status_df.head()
df = pd.DataFrame(np.random.randn(len(status_df), 1))
msk = np.random.rand(len(df)) < 0.6666
status_df_train = status_df[msk]
status_df_test = status_df[~msk]
y_train = status_df_train.bikes_available
status_df_train.drop(columns=["bikes_available"],inplace=True)
y_test = status_df_test.bikes_available
status_df_test.drop(columns=["bikes_available"],inplace=True)
features = [c for c in status_df_train.columns if c not in ['time','installation_date','date','city','lat','long','name']]
features
import time
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(status_df_train))
predictions = np.zeros(len(status_df_test))
start = time.time()
feature_importance_df = pd.DataFrame()


param = {'num_leaves': 100,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': 6,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}
for fold_, (trn_idx, val_idx) in enumerate(folds.split(status_df_train.values, y_train.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(status_df_train.iloc[trn_idx][features], label=y_train.iloc[trn_idx])
    val_data = lgb.Dataset(status_df_train.iloc[val_idx][features], label=y_train.iloc[val_idx])

    num_round = 500
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(status_df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(status_df_test[features], num_iteration=clf.best_iteration) / folds.n_splits
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
