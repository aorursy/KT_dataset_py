import pandas as pd 
import numpy as numpy 
import matplotlib.pyplot as plt
# load data 
ts1 = pd.read_csv('../input/timeseriesvar/timeseries_1.csv') # regular periodic ts
ts2 = pd.read_csv('../input/timeseriesvar/timeseries_2.csv') # irregular ts
# visualize first time series 
plt.figure(figsize = (20, 4))
plt.plot(ts1['nr_people'])
plt.show()

# visualize second time series 
plt.figure(figsize = (20, 4))
plt.plot(ts2['nr_people'])
plt.show()
# percent change of time series 1
percent_change_1 = 100 * (ts1['nr_people'].pct_change())
std_dev_ts1 = percent_change_1.std()


# percent change of time series 2
percent_change_2 = 100 * (ts2['nr_people'].pct_change())
std_dev_ts2 = percent_change_2.std()
print(std_dev_ts1)
print(std_dev_ts2)