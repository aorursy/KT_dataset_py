import numpy as np

import pandas as pd

import seaborn as sns

pd.set_option("max_columns", None)
listings = pd.read_csv("../input/listings.csv")
listings.head()
join_dates = pd.to_datetime(listings['host_since']).value_counts().resample('D').mean().fillna(0)
join_dates.plot()
join_dates.value_counts()
np.argmax(join_dates)
join_dates.rolling(window=31).mean().plot()
pd.to_datetime(listings['host_since']).dt.dayofweek.value_counts().sort_index().plot(kind='bar')
join_dates.resample("M").sum().plot()

join_dates.resample("M").sum().expanding(min_periods=4).mean().plot()