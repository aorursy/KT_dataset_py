def makeTimeSeries(df):
    ts = pd.to_datetime(df.dt)
    df.index = ts
    return df.drop('dt', axis=1)
ts = makeTimeSeries(GlobalTemps)
ts.LandAverageTemperature.plot()
pd.rolling_mean(ts.LandAverageTemperature, 10, freq='A').plot()
# zoom in to that time frame and take a rolling min, max, and mean of the time
yearWithoutSummer = ts['1800':'1850'].LandAverageTemperature
pd.rolling_min(yearWithoutSummer, 24).plot()
pd.rolling_max(yearWithoutSummer, 24).plot()
pd.rolling_mean(yearWithoutSummer, 24).plot()