!pip install tslearn

import tslearn # This module uses sklearn but has API for time series data

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from tslearn.clustering import TimeSeriesKMeans 

from tslearn.neural_network import TimeSeriesMLPClassifier
df = pd.read_csv("../input/worldwide-gdp-history-19602016/gdp_data.csv")

df["time"] = pd.to_datetime(df.Year, format="%Y")

df = df.fillna(0)
countries = list(df.Country.unique())

train_countries = countries[:50]

test_countries = countries[50:60]

train_df = df[df.Country.isin(train_countries)]

test_df = df[df.Country.isin(test_countries)]
train_pivot = train_df.reset_index().pivot_table(index='time', columns='Country', values='GDP-Growth', aggfunc = 'sum')

train_pivot = train_pivot.T

test_pivot = test_df.reset_index().pivot_table(index='time', columns='Country', values='GDP-Growth', aggfunc = 'sum')

test_pivot = test_pivot.T

test_pivot.head()
df_train = np.array(train_pivot).reshape(train_pivot.shape[0],train_pivot.shape[1],1)

df_test = np.array(test_pivot).reshape(test_pivot.shape[0],test_pivot.shape[1],1)

sz = df_train.shape[1]



# DBA-k-means

print("DBA k-means")

dba_km = TimeSeriesKMeans(n_clusters=4,

                          n_init=2,

                          metric="softdtw",

                          verbose=True,

                          random_state=0)

y_pred = dba_km.fit_predict(df_train)

labels = dict(zip(df_pivot.index, y_pred))



#Countries in the first cluster

list(train_pivot[y_pred==0].index)
#Countries in the second cluster

list(train_pivot[y_pred==1].index)
#Countries in the third cluster

list(train_pivot[y_pred==2].index)
#Countries in the fourth cluster

list(train_pivot[y_pred==3].index)
plt.figure(figsize=(18,12))

sns.lineplot(data=pd.DataFrame(train_pivot[y_pred==0].stack()).reset_index(), x="time", y=0, hue="Country")
plt.figure(figsize=(18,12))

sns.lineplot(data=pd.DataFrame(train_pivot[y_pred==1].stack()).reset_index(), x="time", y=0, hue="Country")
plt.figure(figsize=(18,12))

sns.lineplot(data=pd.DataFrame(train_pivot[y_pred==2].stack()).reset_index(), x="time", y=0, hue="Country")
plt.figure(figsize=(18,12))

sns.lineplot(data=pd.DataFrame(train_pivot[y_pred==3].stack()).reset_index(), x="time", y=0, hue="Country")
mlp = TimeSeriesMLPClassifier(hidden_layer_sizes=(64, 64), random_state=0)

mlp.fit(df_train, y_pred)

predictions = mlp.predict(df_test)
predictions
if (predictions==0).any():

    plt.figure(figsize=(18,12))

    sns.lineplot(data=pd.DataFrame(test_pivot[predictions==0].stack()).reset_index(), x="time", y=0, hue="Country")
if (predictions==1).any():

    plt.figure(figsize=(18,12))

    sns.lineplot(data=pd.DataFrame(test_pivot[predictions==1].stack()).reset_index(), x="time", y=0, hue="Country")
if (predictions==2).any():

    plt.figure(figsize=(18,12))

    sns.lineplot(data=pd.DataFrame(test_pivot[predictions==2].stack()).reset_index(), x="time", y=0, hue="Country")
if (predictions==3).any():

    plt.figure(figsize=(18,12))

    sns.lineplot(data=pd.DataFrame(test_pivot[predictions==3].stack()).reset_index(), x="time", y=0, hue="Country")