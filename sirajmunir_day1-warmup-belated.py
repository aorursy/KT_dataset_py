# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/road-weather-information-stations.csv")
df.head()
grouped_data = df.groupby('StationName')
for name, group in grouped_data:
    # Filter only hourly data for display
    df = group[group['DateTime'].str.endswith('10:00')]
    df.loc[:,['DateTime','RoadSurfaceTemperature', 'AirTemperature']].plot(x='DateTime', figsize=(12,8))
    plt.title("Station: {} at 10:00".format(name))