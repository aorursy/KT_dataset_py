import pandas as pd

import numpy as np # linear algebra

from matplotlib import pyplot as plt

import scipy.stats as stats

%matplotlib inline

import seaborn as sns

import numpy as np

import pylab as pl

sns.set(color_codes=True)
with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
print('Dataframe shape:', df.shape)

print('Columns', df.columns.values)
print('Ids (each id represents a different stock, e.g. Apple, Google, etc.) count:', len(df['id'].unique()))

print('Time frames count:', len(df['timestamp'].unique()))
id_count = [len(df[df['timestamp'] == i]['id'].unique()) for i in range(1813)]

plt.figure(figsize=(9,3))

plt.xlabel('Timestamp index')

plt.ylabel('Unique IDs for the timestamp')

plt.plot(range(1813), id_count,'.b')

plt.show()
features = [feature for feature in df.columns.values if not feature in ['id', 'timestamp']]

for feature in features:

    values = df[feature].values

    nan_count = np.count_nonzero(np.isnan(values))

    values = sorted(values[~np.isnan(values)])

    print('NaN count:', nan_count, 'Unique count:', len(np.unique(values)))

    print('Max:', np.max(values), 'Min:', np.min(values))

    print('Median', np.median(values), 'Mean:', np.mean(values), 'Std:', np.std(values))

    plt.figure(figsize=(8,5))

    plt.title('Values '+feature)

    plt.plot(values,'.b')

    plt.show()

    

    plt.figure(figsize=(8,5))

    plt.title('Percentiles 1,5,10...95,99 '+feature)

    percentiles = [1] + list(range(5,100,5)) +[99]

    plt.plot(percentiles, np.percentile(values, percentiles),'.b')

    plt.show()

    

    fit = stats.norm.pdf(values, np.mean(values), np.std(values))  #this is a fitting indeed

    plt.title('Distribution Values '+feature)

    plt.plot(values,fit,'-g')

    plt.hist(values,normed=True, bins=10)      #use this to draw histogram of your data

    plt.show()