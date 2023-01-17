import pandas as pd
%matplotlib inline
# use correct path here
df_april = pd.read_csv('../input/2018_04.csv', index_col=False)
df_april.head(2)
df_april[(df_april["train_id"] == "7837") & (df_april["date"] == "2018-04-01")]
df_april['scheduled_time'] = pd.to_datetime(df_april['scheduled_time'])
df_april['actual_time'] = pd.to_datetime(df_april['actual_time'])
cumu_delay = df_april.groupby(['date', 'train_id']).last()
cumu_delay.head(2)
# Get cumulative delay for NJ Transit trains to New York Penn Station
njt_nyp = cumu_delay[(cumu_delay['type'] == "NJ Transit") & (cumu_delay['to'] == "New York Penn Station")]
njt_nyp.head(2)
njt_nyp[njt_nyp["delay_minutes"] >= 5]["delay_minutes"].hist(bins=20)
# filter based on the "date" index, which level 0 of the multiindex
njt_nyp_0402 = njt_nyp.loc[njt_nyp.index.get_level_values(0) == "2018-04-02"]
njt_nyp_0402.head(2)
njt_nyp_0402.plot(x="scheduled_time", y="delay_minutes", figsize=(8,6))
amtrak = df_april[df_april["type"] == "Amtrak"]
amtrak.head(2)
amtrak[(amtrak['train_id'] == "A2205") & (amtrak['date'] == '2018-04-01')]
# get first and last stops for Amtrak trains
amtrak_first = amtrak.groupby(['date', 'train_id']).first()
amtrak_last = amtrak.groupby(['date', 'train_id']).last()

# calculate total trip times for Amtrak trains
amtrak_trip_times = amtrak_last["actual_time"] - amtrak_first["actual_time"]
amtrak_trip_times.head()
amtrak[amtrak["status"] == "cancelled"].head(5)