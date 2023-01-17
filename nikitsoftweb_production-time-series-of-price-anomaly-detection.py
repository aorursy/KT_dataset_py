# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.dates as md

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import host_subplot

import mpl_toolkits.axisartist as AA

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.covariance import EllipticEnvelope

from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

from mpl_toolkits.mplot3d import Axes3D

# from pyemma import msm

%matplotlib inline



# plt.style.use("fivethirtyeight")





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
expedia = pd.read_csv('/kaggle/input/expedia-personalized-sort/data/train.csv')

df = expedia.loc[expedia['prop_id'] == 104517]

df = df.loc[df['srch_room_count'] == 1]

df = df.loc[df['visitor_location_country_id'] == 219]

df = df[['date_time', 'price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
df.info()
df['price_usd'].describe()
expedia.loc[(expedia['price_usd'] == 5584) & (expedia['visitor_location_country_id'] == 219)]
df = df.loc[df['price_usd'] < 5584]
df.plot(x='date_time', y = 'price_usd', figsize = (20,5))

plt.xlabel('Date time')

plt.ylabel('Price in USD')

plt.title('Time Series of room price by date time of search')
a = df.loc[df['srch_saturday_night_bool'] == 0, 'price_usd']

b = df.loc[df['srch_saturday_night_bool'] == 1, 'price_usd']

plt.figure(figsize=(20, 6))

plt.hist(a, bins = 50, alpha=0.5, label='Search Non-Sat Night')

plt.hist(b, bins = 50, alpha=0.5, label='Search Sat Night')

plt.legend(loc='upper right')

plt.xlabel('Price\n Price is more stable and lower when searching Non-Saturday night and price goes up when searching Saturday night', fontsize = 18)

plt.ylabel('Count', fontsize = 18)

plt.title("Price Comperision between Non-Saturday Night vs Saturday Night")

plt.show();
data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

n_cluster = range(1, 20)

kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]

scores = [kmeans[i].score(data) for i in range(len(kmeans))]



fig, ax = plt.subplots(figsize=(20,6))

ax.plot(n_cluster, scores)

plt.xlabel('Number of Clusters', fontname="Times New Roman",fontweight="bold")

plt.ylabel('Score',fontname="Times New Roman",fontweight="bold")

plt.title("Elbow Curve",fontname="Times New Roman",fontweight="bold")

plt.show();
X = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

X = X.reset_index(drop=True)

km = KMeans(n_clusters=10)

km.fit(X)

km.predict(X)

labels = km.labels_

#Plotting

fig = plt.figure(1, figsize=(12,12))

ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=49, azim=140)

ax.scatter(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2], c=labels.astype(np.float), edgecolor="r")

ax.set_xlabel("price_usd")

ax.set_ylabel("srch_booking_window")

ax.set_zlabel("srch_saturday_night_bool")

plt.title("K Means Clustering for Anomaly Detection", fontsize=20, fontweight="bold");
data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

X = data.values

X_std = StandardScaler().fit_transform(X)

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key = lambda x: x[0], reverse= True)

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance



plt.figure(figsize=(20, 6))

plt.bar(range(len(var_exp)), var_exp, alpha=0.3, align='center', label='individual explained variance', color = 'g')

plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.show();
# Take useful feature and standardize them

data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

X_std = StandardScaler().fit_transform(X)

data = pd.DataFrame(X_std)

# reduce to 2 important features

pca = PCA(n_components=2)

data = pca.fit_transform(data)

# standardize these 2 new features

scaler = StandardScaler()

np_scaled = scaler.fit_transform(data)

data = pd.DataFrame(np_scaled)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]

df['cluster'] = kmeans[9].predict(data)

df.index = data.index

df['principal_feature1'] = data[0]

df['principal_feature2'] = data[1]

df['cluster'].value_counts()
df.head()
# plot the different clusters with the 2 main features

fig, ax = plt.subplots(figsize=(10,6))

colors = {0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7:'yellow', 8:'brown', 9:'purple', 10:'white', 11: 'grey'}

ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df["cluster"].apply(lambda x: colors[x]))

plt.show();
# return Series of distance between each point and its distance with the closest centroid

import sys

def getDistanceByPoint(data, model):

    distance = pd.Series()

    for i in range(0,len(data)):

        Xa = np.array(data.loc[i])

        Xb = model.cluster_centers_[model.labels_[i]-1]

        distance.set_value(i, np.linalg.norm(Xa-Xb))

    return distance



outliers_fraction = 0.01

# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly

distance = getDistanceByPoint(data, kmeans[9])

number_of_outliers = int(outliers_fraction*len(distance))

threshold = distance.nlargest(number_of_outliers).min()

# anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly) 

df['anomaly1'] = (distance >= threshold).astype(int)
fig, ax = plt.subplots(figsize=(20,8))

colors = {0:'blue', 1:'red'}

ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df["anomaly1"].apply(lambda x: colors[x]))

plt.xlabel('principal feature1')

plt.ylabel('principal feature2')

plt.show();
df.anomaly1.value_counts()
df = df.sort_values('date_time')

df["date_time"] = pd.to_datetime(df["date_time"])

df['date_time_int'] = df.date_time.astype(np.int64)

fig, ax = plt.subplots(figsize=(20,6))



a = df.loc[df['anomaly1'] == 1, ['date_time_int', 'price_usd']] #anomaly



ax.scatter(a['date_time_int'],a['price_usd'], color='red', label='Anomaly',s = 200)

ax.plot(df['date_time_int'], df['price_usd'], color='blue', label='Normal',linewidth=0.7)

plt.xlabel('Date Time Integer')

plt.ylabel('price in USD')

plt.legend()

plt.show();
a = df.loc[df['anomaly1'] == 0, 'price_usd']

b = df.loc[df['anomaly1'] == 1, 'price_usd']



fig, axs = plt.subplots(figsize=(20,6))

axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])

plt.show();
import altair as alt

alt.renderers.enable('default')

data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

scaler = StandardScaler()

np_scaled = scaler.fit_transform(data)

data = pd.DataFrame(np_scaled)

# train isolation forest

model =  IsolationForest(contamination=outliers_fraction)

model.fit(data)



df['anomaly2'] = pd.Series(model.predict(data))

# df['anomaly2'] = df['anomaly2'].map( {1: 0, -1: 1} )



fig, ax = plt.subplots(figsize=(20,10))



a = df.loc[df['anomaly2'] == -1, ['date_time_int', 'price_usd']] #anomaly



ax.plot(df['date_time_int'], df['price_usd'], color='blue', label = 'Normal',linewidth=0.7)

ax.scatter(a['date_time_int'],a['price_usd'], color='red', label = 'Anomaly', s = 200)

plt.legend()

plt.show();
# visualisation of anomaly with avg price repartition

a = df.loc[df['anomaly2'] == 1, 'price_usd']

b = df.loc[df['anomaly2'] == -1, 'price_usd']



fig, axs = plt.subplots(figsize=(20,8))

axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])

plt.show();
data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

scaler = StandardScaler()

np_scaled = scaler.fit_transform(data)

data = pd.DataFrame(np_scaled)

# train oneclassSVM 

model = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01)

model.fit(data)

 

df['anomaly3'] = pd.Series(model.predict(data))

# df['anomaly3'] = df['anomaly3'].map( {1: 0, -1: 1} )

fig, ax = plt.subplots(figsize=(20,6))



a = df.loc[df['anomaly3'] == -1, ['date_time_int', 'price_usd']] #anomaly



ax.plot(df['date_time_int'], df['price_usd'], color='blue', label ='Normal', linewidth = 0.7)

ax.scatter(a['date_time_int'],a['price_usd'], color='red', label = 'Anomaly', s = 100)

plt.legend()

plt.show();
a = df.loc[df['anomaly3'] == 1, 'price_usd']

b = df.loc[df['anomaly3'] == -1, 'price_usd']



fig, axs = plt.subplots(figsize=(20,6))

axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])

plt.show();
df_class0 = df.loc[df['srch_saturday_night_bool'] == 0, 'price_usd']

df_class1 = df.loc[df['srch_saturday_night_bool'] == 1, 'price_usd']



fig, axs = plt.subplots(1,2, figsize= (20,5))

df_class0.hist(ax=axs[0], bins=30)

df_class1.hist(ax=axs[1], bins=30)

axs[0].set_title("Non Saturday Night")

axs[1].set_title("Saturday Night")
envelope =  EllipticEnvelope(contamination = outliers_fraction) 

X_train = df_class0.values.reshape(-1,1)

envelope.fit(X_train)

df_class0 = pd.DataFrame(df_class0)

df_class0['deviation'] = envelope.decision_function(X_train)

df_class0['anomaly'] = envelope.predict(X_train)



envelope =  EllipticEnvelope(contamination = outliers_fraction) 

X_train = df_class1.values.reshape(-1,1)

envelope.fit(X_train)

df_class1 = pd.DataFrame(df_class1)

df_class1['deviation'] = envelope.decision_function(X_train)

df_class1['anomaly'] = envelope.predict(X_train)
# plot the price repartition by categories with anomalies

a0 = df_class0.loc[df_class0['anomaly'] == 1, 'price_usd']

b0 = df_class0.loc[df_class0['anomaly'] == -1, 'price_usd']



a2 = df_class1.loc[df_class1['anomaly'] == 1, 'price_usd']

b2 = df_class1.loc[df_class1['anomaly'] == -1, 'price_usd']



fig, axs = plt.subplots(1,2, figsize= (20,5))

axs[0].hist([a0,b0], bins=32, stacked=True, color=['blue', 'red'])

axs[1].hist([a2,b2], bins=32, stacked=True, color=['blue', 'red'])

axs[0].set_title("Search Non Saturday Night")

axs[1].set_title("Search Saturday Night")

plt.show();
# add the data to the main 

df_class = pd.concat([df_class0, df_class1])

df['anomaly5'] = df_class['anomaly']

# df['anomaly5'] = np.array(df['anomaly22'] == -1).astype(int)

fig, ax = plt.subplots(figsize=(20, 6))

a = df.loc[df['anomaly5'] == -1, ('date_time_int', 'price_usd')] #anomaly

ax.plot(df['date_time_int'], df['price_usd'], color='blue', label='Normal', linewidth = 0.7)

ax.scatter(a['date_time_int'],a['price_usd'], color='red', label='Anomaly', s = 100)

plt.legend()

plt.show();