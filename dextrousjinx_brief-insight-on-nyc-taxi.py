import os

import math

import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline

from mpl_toolkits.basemap import Basemap, cm

import matplotlib.pyplot as plt

for f in os.listdir('../input'):

    size_bytes = round(os.path.getsize('../input/' + f)/ 1000, 2)

    size_name = ["KB", "MB"]

    i = int(math.floor(math.log(size_bytes, 1024)))

    p = math.pow(1024, i)

    s = round(size_bytes / p, 2)

    print(f.ljust(25) + str(s).ljust(7) + size_name[i])
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train Data".ljust(15), train_df.shape)

print("Test Data".ljust(15), train_df.shape)
print(train_df.head())
print(test_df.head())
sns.set(style="white")

corr = train_df.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
print(train_df["passenger_count"].unique())
plt.figure(figsize=(12,8))

sns.countplot(x="passenger_count", data=train_df)

#sns.swarmplot(x="passenger_count", y="trip_duration", data=train_df, color="w", alpha=.5);
plt.figure(figsize=(12,8))

sns.countplot(x="passenger_count", data=train_df[train_df["passenger_count"].isin([0,7,8,9])])
some = train_df[train_df["passenger_count"].isin([7,8,9])]

print(some["vendor_id"].unique())
plt.figure(figsize=(12,8))

sns.countplot(x="vendor_id", data=train_df)
plt.figure(figsize=(12,8))

sns.countplot(x="passenger_count", hue="vendor_id", data=train_df);
plt.figure(figsize=(12,8))

sns.countplot(x="passenger_count", hue="vendor_id", data=train_df[train_df["passenger_count"]==0]);
plt.figure(figsize=(12,8))

sns.lmplot(x="pickup_longitude", y="pickup_latitude", hue="vendor_id", data=train_df, fit_reg=False);