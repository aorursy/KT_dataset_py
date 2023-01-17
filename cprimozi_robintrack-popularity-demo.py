import pandas as pd



import os

from datetime import datetime



tsla_df = pd.read_csv("../input/tmp/popularity_export/TSLA.csv")

tsla_df['timestamp'] = tsla_df['timestamp'].map(lambda dt_string: datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S'))



# The following is based off of https://stackoverflow.com/a/42372823/3833068

import matplotlib.pyplot as plt

import matplotlib.dates as mdates



# plot

plt.plot(tsla_df['timestamp'], tsla_df['users_holding'])

plt.title('Tesla (TSLA) Popularity over Time')

plt.ylabel('Unique Robinhood Users Holding Shares')

# beautify the x-labels

plt.gcf().autofmt_xdate()



plt.show()