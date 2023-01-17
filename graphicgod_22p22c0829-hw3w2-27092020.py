import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import numpy as np
# import data
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df['neighbourhood_group'].value_counts()
# Encode
df['neighbourhood'].value_counts()
# too much unigue value --> drop this column
df['room_type'].value_counts()
# Encode
# Encode neighbourhood_group
one_hot = pd.get_dummies(df['neighbourhood_group'], prefix='neighbourhood_group')
df.drop('neighbourhood_group', axis=1, inplace=True)
df = df.join(one_hot)
df.head()
# Encode room_type columns
one_hot = pd.get_dummies(df['room_type'], prefix='room_type')
df.drop('room_type', axis=1, inplace=True)
df = df.join(one_hot)
df.head()
# Drop categorical columns
df.drop(['name', 'host_id', 'host_name','neighbourhood','last_review'], axis=1, inplace=True)
df.dropna(inplace=True)

df = StandardScaler().fit_transform(df)
df
# Plot Dendogram
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(30, 7))
plt.title("Dendograms")
plt.xlabel('Customer')
plt.ylabel('Distance')
dend = shc.dendrogram(shc.linkage(df, method='ward'))
