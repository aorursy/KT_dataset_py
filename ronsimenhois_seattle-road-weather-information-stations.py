import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("../input/road-weather-information-stations.csv")
data.head()
# Break the data into the different weather stations
grouped = data.groupby('StationName')
for name, group in grouped:
    print('Station name {0}:'.format(name))
    print(group.iloc[:,-2:].describe())
    # Filter only hourly data for display
    df = group[group['DateTime'].str.endswith('00:00')]
    df.loc[:,['DateTime','RoadSurfaceTemperature', 'AirTemperature']].plot(x='DateTime', figsize=(12,8))
    plt.title("Station: {} at 12:00".format(name))

