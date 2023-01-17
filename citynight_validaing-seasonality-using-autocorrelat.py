import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import acf
df_train = pd.read_csv("../input/train.csv", dtype={"place_id": object})

# assume time is minute
df_train['hour'] = df_train.time // 60
df_train['day'] = df_train.time // (60*24)

# use the place id with the most checkins as an example
checkins_by_place = df_train.place_id.value_counts()
df_sample_place = df_train[df_train.place_id == checkins_by_place.index[0]]
# this function plots the ACF when you input frequency series with time index
def check_acf(counts_by_time):

    # fill in the gap with 0 to create a series with fixed intervals
    time_index = np.arange(
        counts_by_time.index.min(),
        counts_by_time.index.max() + 1)
    count_by_time_filled = counts_by_time.reindex(time_index)
    count_by_time_filled.fillna(0, inplace=True)

    # ACF
    acf_raw = acf(count_by_time_filled)
    
    # plot
    sns.barplot(x=np.arange(0, acf_raw.size), y=acf_raw)
# plot the ACF of "day"
check_acf(df_sample_place.day.value_counts())
# plot the ACF of hour
check_acf(df_sample_place.hour.value_counts())