import pandas as pd
cab_df = pd.read_csv("../input/cab_rides.csv")#,encoding = "utf-16")
weather_df = pd.read_csv("../input/weather.csv")#,encoding = "utf-16")
cab_df.head()
weather_df.head()

cab_df['date_time'] = pd.to_datetime(cab_df['time_stamp']/1000, unit='s')
weather_df['date_time'] = pd.to_datetime(weather_df['time_stamp'], unit='s')
cab_df.head()
weather_df.head()
#merge the datasets to refelect same time for a location
cab_df['merge_date'] = cab_df.source.astype(str) +" - "+ cab_df.date_time.dt.date.astype("str") +" - "+ cab_df.date_time.dt.hour.astype("str")
weather_df['merge_date'] = weather_df.location.astype(str) +" - "+ weather_df.date_time.dt.date.astype("str") +" - "+ weather_df.date_time.dt.hour.astype("str")
weather_df.index = weather_df['merge_date']
cab_df.head()
merged_df = cab_df.join(weather_df,on=['merge_date'],rsuffix ='_w')
print(merged_df.shape)
merged_df['rain'].fillna(0,inplace=True)
merged_df = merged_df[pd.notnull(merged_df['date_time_w'])]
print(merged_df.shape)
merged_df = merged_df[pd.notnull(merged_df['price'])]
print(merged_df.shape)
merged_df['day'] = merged_df.date_time.dt.dayofweek
merged_df['hour'] = merged_df.date_time.dt.hour
merged_df['day'].describe()
merged_df.columns
merged_df.count()
merged_df["price_div_distance"] = merged_df["price"].div(merged_df["distance"])
merged_df
merged_df = merged_df.drop(["id","merge_date","date_time","merge_date_w","time_stamp_w"]).drop_duplicates().sample(frac=1)
merged_df.to_csv("-uber-lyft-ride-prices.csv.gz",index=False,compression="gzip")