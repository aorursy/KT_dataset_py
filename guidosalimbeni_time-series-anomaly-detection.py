import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df_daily = pd.read_csv("/kaggle/input/bike-share-daily-data/bike_sharing_daily.csv")

df_hourly = pd.read_csv("/kaggle/input/bike-share-daily-data/bike_sharing_hourly.csv")
df = [df_daily,df_hourly]

for d in df:

    print (d.isnull().any().sum())
def fixing_datatypes(df):

    # Fixing the datatypes 

    df['dteday'] = df['dteday'].astype('datetime64')

    df.loc[:,'season':'mnth'] = df.loc[:,'season':'mnth'].astype('category')

    df[['holiday','workingday']] = df[['holiday','workingday']].astype('bool')

    df[['weekday','weathersit']] = df[['weekday','weathersit']].astype('category')



    

      

    return df
df_daily = fixing_datatypes(df_daily)

df_hourly = fixing_datatypes(df_hourly)



df_hourly['hr'] = df_hourly['hr'].astype('category')
# set the index to datetime

df_daily = df_daily.set_index('dteday')
df_daily["cnt"].plot(figsize = (40,10))

df_daily_resample = df_daily.resample(rule = "M").mean().ffill()

df_daily_resample["cnt"].plot(figsize = (40,10))
df_daily_resample["cnt"].resample("M").mean().plot.bar(figsize = (40,10))
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df_daily_resample["cnt"], model = "multiplicative")

fig = result.plot()
df_daily.head()
sns.boxplot(df_daily['mnth'], df_daily['cnt'])
df_daily[df_daily.cnt == df_daily.cnt.min()]
df_daily[df_daily.cnt == df_daily.cnt.max()]
from sklearn.cluster import KMeans


df_daily = df_daily.reset_index()

df_daily.head()
# subset of the daily data for the k-means anomaly detection test

df = df_daily[["cnt", "season", 'yr', 'mnth', 'holiday', 'weekday', 'workingday',

       'weathersit', 'atemp', 'hum', 'windspeed']]
# check the correlation of features of this subset 

sns.heatmap(abs(df.corr()), annot = True)
from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()

np_scaled = Scaler.fit_transform(df)

df = pd.DataFrame(np_scaled)

df.head()
from sklearn.decomposition import PCA

# reduce to 2 importants features

pca = PCA(n_components=2)

df = pca.fit_transform(df)

# standardize these 2 new features

min_max_scaler = StandardScaler()

np_scaled = min_max_scaler.fit_transform(df)

df = pd.DataFrame(np_scaled)
df.head()
# calculate with different number of centroids to see the loss plot (elbow method)

n_cluster = range(1, 20)

kmeans = [KMeans(n_clusters=i).fit(df) for i in n_cluster]

scores = [kmeans[i].score(df) for i in range(len(kmeans))]

fig, ax = plt.subplots()

ax.plot(n_cluster, scores)

plt.show()
SelectedKey = 3
df_daily['cluster'] = kmeans[SelectedKey].predict(df)

df_daily['principal_feature1'] = df[0]

df_daily['principal_feature2'] = df[1]

df_daily['cluster'].value_counts().plot.bar()
df_daily.head()
#plot the different clusters with the 2 main features



sns.scatterplot(df_daily['principal_feature1'], df_daily['principal_feature2'], hue=df_daily["cluster"], data = df_daily, style = df_daily["cluster"])

# return Series of distance between each point and his distance with the closest centroid

def getDistanceByPoint(data, model):

    distance = pd.Series()

    for i in range(0,len(data)):

        Xa = np.array(data.loc[i])

        Xb = model.cluster_centers_[model.labels_[i]-1]

        distance.set_value(i, np.linalg.norm(Xa-Xb))

    return distance
outliers_fraction = 0.1
# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly

distance = getDistanceByPoint(df, kmeans[SelectedKey])

number_of_outliers = int(outliers_fraction*len(distance))

threshold = distance.nlargest(number_of_outliers).min() #Return the first n rows ordered by columns in descending order.

# anomaly21 contain the anomaly result of method 2.1 Cluster (0:normal, 1:anomaly) 

df_daily['anomaly21'] = (distance >= threshold).astype(int)
fig, ax = plt.subplots(figsize=(10,10))

sns.scatterplot(df_daily['principal_feature1'], df_daily['principal_feature2'], hue=df_daily['anomaly21'], data = df_daily, style = df_daily["cluster"], ax = ax)

plt.show()
# set the index to datetime

df_daily = df_daily.set_index('dteday')
df_daily["cnt"].plot(figsize = (20,10))

plt.scatter (df_daily.index[df_daily['anomaly21'] == 1], df_daily["cnt"][df_daily['anomaly21'] == 1], c = "red")
anomalies = df_daily[df_daily['anomaly21'] == 1]

anomalies.loc['2012-10-1':'2012-12-31'][anomalies["cnt"] < 500]
sns.barplot(x = df_daily["season"] , y = df_daily["atemp"], hue = df_daily["anomaly21"])
sns.pairplot(df_daily[[ 'anomaly21', 'atemp', 'casual', 'hum']] , hue = "anomaly21")
df_daily = df_daily.reset_index()
# subset of the daily data for the k-means anomaly detection test

df = df_daily[["cnt", "season", 'yr', 'mnth', 'holiday', 'weekday', 'workingday',

       'weathersit', 'atemp', 'hum', 'windspeed']]

scaler = StandardScaler()

np_scaled = scaler.fit_transform(df)

df = pd.DataFrame(np_scaled)
from sklearn.ensemble import IsolationForest

model =  IsolationForest(contamination = outliers_fraction)

model.fit(df)

df_daily['anomaly_isolation'] = pd.Series(model.predict(df))

df_daily['anomaly_isolation'] = df_daily['anomaly_isolation'].map( {1: 0, -1: 1} )

print(df_daily['anomaly_isolation'].value_counts())
 # set the index to datetime

df_daily = df_daily.set_index('dteday')

df_daily["cnt"].plot(figsize = (20,10))

plt.scatter (df_daily.index[df_daily['anomaly_isolation'] == 1], df_daily["cnt"][df_daily['anomaly_isolation'] == 1], c = "red")
df_daily = df_daily.reset_index()

# subset of the daily data for the k-means anomaly detection test

df = df_daily[["cnt", "season", 'yr', 'mnth', 'holiday', 'weekday', 'workingday',

       'weathersit', 'atemp', 'hum', 'windspeed']]

scaler = StandardScaler()

np_scaled = scaler.fit_transform(df)

df = pd.DataFrame(np_scaled)

df.shape
# important parameters and train/test size

prediction_time = 1 

testdatasize = 100

unroll_length = 22

testdatacut = testdatasize + unroll_length  + 1



#train data

x_train = df[0:-prediction_time-testdatacut].as_matrix()

y_train = df[prediction_time:-testdatacut  ][0].as_matrix()



# test data

x_test = df[0-testdatacut:-prediction_time].as_matrix()

y_test = df[prediction_time-testdatacut:  ][0].as_matrix()



x_train.shape, y_train.shape, x_test.shape, y_test.shape
def unroll(data,sequence_length):

    result = []

    for index in range(len(data) - sequence_length):

        result.append(data[index: index + sequence_length])

    return np.asarray(result)



# adapt the datasets for the sequence data shape

x_train = unroll(x_train,unroll_length)

x_test  = unroll(x_test,unroll_length)

y_train = y_train[-x_train.shape[0]:]

y_test  = y_test[-x_test.shape[0]:]



# see the shape

print("x_train", x_train.shape)

print("y_train", y_train.shape)

print("x_test", x_test.shape)

print("y_test", y_test.shape)
from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import LSTM

from keras.models import Sequential

model = Sequential()



model.add(LSTM(input_dim=x_train.shape[-1],output_dim=unroll_length,return_sequences=True))

model.add(Dropout(0.2))





model.add(LSTM(100,return_sequences=False))

model.add(Dropout(0.2))



model.add(Dense(units=1))

model.add(Activation('linear'))



model.compile(loss='mse', optimizer='rmsprop')

model.summary()
model.fit(x_train,y_train,batch_size=5,nb_epoch=30,validation_split=0.1)


diff=[]

ratio=[]

p = model.predict(x_test)



for u in range(len(y_test)):

    pr = p[u][0]

    ratio.append((y_test[u]/pr)-1)

    diff.append(abs(y_test[u]- pr))

fig, axs = plt.subplots()

axs.plot(p,color='red', label='prediction')

axs.plot(y_test,color='blue', label='y_test')

plt.legend(loc='upper right')

plt.show()
diff = pd.Series(diff)

number_of_outliers = int(outliers_fraction*len(diff))

threshold = diff.nlargest(number_of_outliers).min()

# data with anomaly label (test data part)

test = (diff >= threshold).astype(int)

# the training data part where we didn't predict anything (overfitting possible): no anomaly

complement = pd.Series(0, index=np.arange(len(df)-testdatasize))

# # add the data to the main

df_daily['anomalyLSTM'] = complement.append(test, ignore_index='True')

print(df_daily['anomalyLSTM'].value_counts())
# set the index to datetime

df_daily = df_daily.set_index('dteday')

df_daily["cnt"].plot(figsize = (20,10))

plt.scatter (df_daily.index[df_daily['anomalyLSTM'] == 1], df_daily["cnt"][df_daily['anomalyLSTM'] == 1], c = "red")