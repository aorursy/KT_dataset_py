import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.describe()
df_test.describe()
x = df_train[df_train.time < 20000]
y = x[x.time > 10000]
y['x'].plot.hist(bins=50)
ax = df_train[df_train.place_id == 8772469670].plot.scatter(x='time',y='y',c='red', s=0.1, figsize=(18,6))
df_train[df_train.place_id == 1623394281].plot.scatter(x='time',y='y',c='green', ax=ax,s=0.1)
df_train[df_train.place_id == 1308450003].plot.scatter(x='time',y='y',c='black', ax=ax,s=0.1)
df_train[df_train.place_id == 4823777529].plot.scatter(x='time',y='y',c='yellow', ax=ax,s=0.1)
df_train[df_train.place_id == 9586338177].plot.scatter(x='time',y='y',c='blue', ax=ax,s=0.1)

#df_train.groupby(['place_id']).agg(['count']).describe()
#df_train.groupby(['place_id'])['place_id'].agg(['count']).sort(['count'], ascending=[0]).head(10)
df_train.groupby(['place_id'])['place_id'].agg(['count'])['count'].plot.hist(bins=300, xlim=(0,100))
df_train[df_train['place_id'].isin([5351837004]) ]['time'].apply(lambda x: x%(60*24)).plot.hist(bins = 24)