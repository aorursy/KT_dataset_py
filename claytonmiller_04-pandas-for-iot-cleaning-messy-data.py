import pandas as pd
rawdata = pd.read_csv("../input/weather0.csv", index_col='timestamp', parse_dates=True)
rawdata.head()
rawdata.info()
rawdata_hourly = rawdata.resample("H").mean()
rawdata_hourly.info()
rawdata_hourly.plot(figsize=(20,10))
rawdata_hourly.plot(figsize=(20,15), subplots=True)
rawdata_hourly_nooutliers = rawdata_hourly[rawdata_hourly > -40]
rawdata_hourly_nooutliers.plot(figsize=(20,15), subplots=True)
weatherfilelist = ["weather0.csv","weather4.csv","weather8.csv","weather10.csv"]
temp_data = []
for weatherfilename in weatherfilelist:
    print("Getting data from: "+weatherfilename)
    
    rawdata = pd.read_csv("../input/"+weatherfilename, index_col='timestamp', parse_dates=True)
    rawdata_hourly = rawdata.resample("H").mean()
    rawdata_hourly_nooutliers = rawdata_hourly[rawdata_hourly > -40]
    
    temperature = rawdata_hourly_nooutliers["TemperatureC"]
    temperature.name = weatherfilename
    
    temp_data.append(temperature)
all_temp_data = pd.concat(temp_data, axis=1)
all_temp_data.head()
all_temp_data.info()
all_temp_data.plot(figsize=(20,10))
all_temp_data.boxplot(vert=False)
all_temp_data.plot.hist(figsize=(20,7), bins=50, alpha=0.5)
