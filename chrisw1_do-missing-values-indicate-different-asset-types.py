import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

cmap=plt.cm.get_cmap('Blues')
# read in the data

with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
# remove ids that do not have an entry for every timestamp

id_counts = df.id.value_counts()

ids = np.setdiff1d(df.id.unique(), id_counts[id_counts != len(df.timestamp.unique())].index)

enduring = df[df.id.isin(ids)]
print('Number of instruments in the full training set:',len(df.id.unique()))

print('Number of instruments in the reduced training set:',len(enduring.id.unique()))
def GetNaFraction(df):

    # form n x m dataframe - one row per id, one column per feature

    # each entry counts the fraction of NAs for the feature for one asset

    nas = pd.DataFrame(index = df.id.unique(), columns=df.columns).T

    for id in df.id.unique():

        nas[id] = df[df.id == id].isnull().mean()

    return nas.T
# Calculate the fraction of NAs per instrument per feature

enduring_nas = GetNaFraction(enduring)

enduring_nas.drop(['id', 'timestamp', 'y'], axis=1, inplace=True)

overall_percentages = enduring_nas.mean(axis=0).sort_values(ascending=False)

enduring_nas.head()
# Plot an overview

fig, ax = plt.subplots()

idx = np.arange(len(overall_percentages))

ax.barh(idx, overall_percentages, color='cyan', ecolor='black')

ax.set_xlabel("Average fraction of missing values")

ax.set_title("Fraction of missing values in each feature")

plt.margins(0.01)

plt.xlim(xmin=0);
no_na_ids = overall_percentages[overall_percentages==0].index

print(sorted(no_na_ids))
constant_na_ids = enduring_nas.columns[enduring_nas.apply(pd.Series.nunique) == 1]

constant_na_ids = np.setdiff1d(constant_na_ids, no_na_ids)

print(constant_na_ids )
constant_nas=enduring[np.append('timestamp', constant_na_ids )].copy()

for col in constant_nas.columns[1:]:

    constant_nas.loc[:,col] = constant_nas[col].isnull().astype(int)



# get the average number of nas per timestamp

constant_nas = constant_nas.groupby(['timestamp']).mean()



# sort the columns shorter->longer moving averages

new_cols = list(constant_nas.apply(sum, axis=0).sort_values(ascending=True).index)



fig = plt.figure()

plt.gca().invert_yaxis()   # flip so that older timestamps are at the top

plt.pcolormesh(constant_nas[new_cols], cmap=cmap)

plt.colorbar(shrink=0.5)

plt.title("Fraction of NAs per Feature per Timestamp")

plt.ylabel("Timestamp")

plt.xlabel("Feature")

plt.axis('tight');
fig = plt.figure(figsize=(8, 8))

plt.pcolormesh(enduring_nas,cmap=cmap) 

plt.colorbar(shrink=0.5)

plt.title("Fraction of NAs per Feature per Instrument")

plt.ylabel("Instruments")

plt.xlabel("Features")

plt.axis('tight');
# resort the matrix to show groupings more strongly

cols = overall_percentages.index.tolist()

cols_to_sort=['fundamental_6', 'fundamental_24']

fig = plt.figure(figsize=(8, 8))

plt.pcolormesh(enduring_nas[cols].sort_values(cols_to_sort),cmap=cmap)

plt.colorbar(shrink=0.5)

plt.title("Fraction of NAs per Feature per Instrument (sorted)")

plt.ylabel("Instruments (sorted)")

plt.xlabel("Features (sorted)")

plt.axis('tight');
# plot distribution of the y-values

# Assume that missing values are dominated by the systematic effects

enduring.y[enduring.fundamental_6.isnull()].plot.kde(color='Blue', label='f6 null')

enduring.y[enduring.fundamental_24.isnull()].plot.kde(color='Red', label='f24 null')

enduring.y[enduring.fundamental_6.notnull() & enduring.fundamental_24.notnull()].plot.kde(color='Orange', label='f6,f24 not null')

plt.title("Density of Y-Values for Different Groupings")

plt.xlabel("Y")

plt.legend();
enduring.y[enduring.fundamental_5.notnull()].plot.kde(color='Red', label='f5 not null')

enduring.y[enduring.fundamental_5.isnull()].plot.kde(color='Blue', label='f5 null')

plt.title("Density of Y-Values: 'fundamental_5' Missing/Not Missing")

plt.xlabel("Y")

plt.legend();