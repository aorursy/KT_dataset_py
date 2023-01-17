import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

%matplotlib inline

plt.rcParams["figure.figsize"] = 10, 6
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

combined_df = pd.concat((train_df, test_df))

# reindex the combined dataframe

combined_df = combined_df.reset_index()

combined_df.drop("index", axis=1, inplace=True)
print(train_df.shape, test_df.shape, combined_df.shape)

combined_df.count()
train_df.head()
test_df.head()
start = dt.datetime.strptime(train_df.loc[0].pickup_datetime, "%Y-%m-%d %X")

end = dt.datetime.strptime(train_df.loc[0].dropoff_datetime, "%Y-%m-%d %X")

print((end-start).total_seconds(), train_df.loc[0].trip_duration)
train_df.store_and_fwd_flag.value_counts()
train_df.trip_duration.describe()
print("Longest trip took {} days.".format(train_df.trip_duration.max()/(60.*60*24)))
max_entry = train_df.where(train_df.trip_duration == train_df.trip_duration.max()).dropna()

max_entry.index
train_df.drop(max_entry.index).describe()
long_durations = train_df.where(train_df.trip_duration > 4e+4).dropna()

long_durations.describe()
train_df.drop(long_durations.index, inplace=True)

train_df.describe()
train_df.hist(column="trip_duration", bins=100)
combined_df.vendor_id.value_counts()
vendor_counts = combined_df.vendor_id.value_counts(sort=True, ascending=True)

df = pd.DataFrame(vendor_counts)

df.columns = ["counts"]

df.plot(kind="bar", stacked=True)
n, bins, patches = plt.hist([train_df[train_df.vendor_id == 1].trip_duration,

                            train_df[train_df.vendor_id == 2].trip_duration],

                            stacked=True, edgecolor="k", bins=1000)

plt.legend(patches, ("Vendor 1", "Vendor 2"), loc="best")

plt.xlim(0., 10000)

plt.show()

n, bins, patches = plt.hist([train_df[(train_df.vendor_id == 1) & (train_df.trip_duration > 10000)].trip_duration,

                            train_df[(train_df.vendor_id == 2) & (train_df.trip_duration > 10000)].trip_duration],

                            stacked=True, edgecolor="k", bins=1000)

plt.legend(patches, ("Vendor 1", "Vendor 2"), loc="best")

plt.show()
# Look at the whole distribution

sns.kdeplot(train_df[train_df.vendor_id == 1].trip_duration, label="Vendor 1", shade=True)

sns.kdeplot( train_df[train_df.vendor_id == 2].trip_duration, label="Vendor 2", shade=True)

plt.xlim(0., 10000)

plt.show()



# Look only at trips < 10000 s

sns.kdeplot(train_df[train_df.vendor_id == 1].trip_duration, label="Vendor 1", shade=True, clip=[0., 10000])

sns.kdeplot( train_df[train_df.vendor_id == 2].trip_duration, label="Vendor 2", shade=True, clip=[0., 10000])

plt.xlim(0., 10000)

plt.show()
combined_df["year"] =  combined_df.pickup_datetime.map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %X").year)

train_df["year"] =  train_df.pickup_datetime.map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %X").year)

combined_df.head()