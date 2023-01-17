import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_csv("../input/weather_data_nyc_centralpark_2016(1).csv")

data
data.describe()