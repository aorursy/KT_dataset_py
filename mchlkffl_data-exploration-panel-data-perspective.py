import pandas as pd

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', 500)

pd.options.display.max_rows = 1000
hdf = pd.read_hdf('../input/train.h5')

print(len(hdf["id"].unique()))

hdf.head()
n_samples = 200

df_sample = pd.DataFrame({"sample_id": pd.Series(hdf["id"].unique()).sample(n_samples) })

df_sample = df_sample.reset_index(drop=True).reset_index(drop=False)

df_sample["group"] = df_sample["index"].apply(lambda x: int((x+1)/10)) # Create groups of 10 sample ids.

hdf = pd.merge(left = hdf, right = df_sample[["sample_id", "group"]], left_on="id", right_on="sample_id", how = "left")

df_sample.head()
# Look at "y" for a random subset of n_samples ids

hdf_plot = hdf.loc[~hdf["sample_id"].isnull(), ["sample_id", "timestamp", "y"]]

grid = sns.FacetGrid( hdf_plot, col = "sample_id", col_wrap=5)

grid = (grid.map(plt.plot, "timestamp", "y")).add_legend()

plt.show()
# Look at all feature columns for 10 sampled ids

sample_id_group = 1

cols_included = list(hdf)[1:]

hdf_melt = hdf.loc[ (~hdf["sample_id"].isnull()) & (hdf["group"] == sample_id_group), cols_included]

hdf_melt = pd.melt (hdf_melt, id_vars=["timestamp","sample_id"])

hdf_melt.head()
grid = sns.FacetGrid( hdf_melt, col = "variable", col_wrap=5, hue  ="sample_id", ylim = [-2, 2]) # useful to clip values as suggested by other people

grid = (grid.map(plt.plot, "timestamp", "value")).add_legend()

plt.show()
# Set an index with two dimensions, id and timestamp. Useful for subsequent computations

hdf = hdf.set_index(["id", "timestamp"])

hdf = hdf.sort_index(level = 1).sort_index(level = 0)



# Number of periods available by id

t="Number of periods available by id"

res = hdf.groupby(level = 0)["y"].apply(lambda x: len(x[~x.isnull()])) # Note: some ids have null rows that would be counted if not removed

res.plot(kind = 'hist', bins = 100, title = t)

plt.show()
res = hdf.groupby(level = 0)["y"].apply(lambda x: x.index.max()[1] )

res.plot(kind = 'hist', bins = 100, title = "Maximum time period for 'y' column by id")

plt.show()
# How about ids that end before 1812, do many of them end early on?

res[res < 1812].plot(kind = 'hist', bins = 4)

plt.show()
def number_empty_columns(df):

    df_nulls = df.apply(lambda col: (pd.isnull(col)*1))

    df_nulls = df_nulls.sum(axis = 0)

    count = ((df_nulls == 0)*1).sum()

    return count



res = hdf.groupby(level = 0).apply(number_empty_columns)

res.plot(kind = 'hist', bins = 100, title = "Number of empty columns by id")

plt.show()
res.hist(cumulative=True, normed=1, bins=10)

plt.suptitle( "Number of empty columns by id - cumulative % frequency")

plt.show()
def average_peridods_by_col(df):

    df_nulls = df.apply(lambda col: (pd.isnull(col)*1)) # returns 1 if value is null, otherwise 0

    df_nulls = df_nulls.sum(axis = 0) #return count of non null values by column

    mapping = np.where(df_nulls > 0, '>0',  '0')

    mean = df_nulls.groupby(mapping).mean() # returns average number of time periods by category

    mean = mean['>0']

    return mean



res2 = hdf.groupby(level=0).apply(average_peridods_by_col)

res2.plot(kind = 'hist', bins = 10, title = "Average number of time periods for non empty columns" )

plt.show()
target_ids = res[res<60].index

mapping = np.in1d(res2.index, target_ids) # return True if id has less than 60 empty columns

mapping = np.where(mapping, "target", "not target")
fig, axes = plt.subplots(nrows=1, ncols=2)

fig.suptitle("Average number of time periods for non empty columns", fontsize=14)

title1, title2  = "Less than 60 empty columns",  "More than 60 empty columns"

res2[res2.index.isin(target_ids)].plot(kind = 'hist', bins = 10, ax=axes[0], sharex =True,sharey =True, title=title1, xlim = (0,1400))

res2[~res2.index.isin(target_ids)].plot(kind = 'hist', bins = 10, ax=axes[1], sharex =True,sharey =True,title=title2, xlim = (0,1400))

plt.show()