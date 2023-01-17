import numpy as np

import pandas as pd 



import seaborn as sns 

import matplotlib.dates as md

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import host_subplot

import mpl_toolkits.axisartist as AA

plt.style.use(['fivethirtyeight', 'dark_background'])



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.covariance import EllipticEnvelope

from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

from mpl_toolkits.mplot3d import Axes3D



!pip install pyemma

from pyemma import msm

%matplotlib inline



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/expedia-personalized-sort/data/train.csv')
train.head()
train.isnull().sum()
# prop_id corresponding to 

train['prop_id'].value_counts()
train['visitor_location_country_id'].value_counts()
# Num of rooms specified in search by customer

train['srch_room_count'].value_counts()
# Subset df 

df = train.loc[train['prop_id'] == 104517]



df = df.loc[df['visitor_location_country_id'] == 219]



df = df.loc[df['srch_room_count'] == 1]



# srch_saturday = if stay includes Sat night 

# srch_booking_window = num of days between search date and hotel stay start date 

df = df[['date_time', 'price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
df.info()
df.describe()
train.loc[(train['price_usd'] == 5584) & 

         (train['visitor_location_country_id'] == 219)]
# Remove 5584 

df = df.loc[df['price_usd'] < 5584]

df['price_usd'].describe()
df['date_time'].min(), df['date_time'].max()
df['date_time'].describe()



df['date_time'] = pd.to_datetime(df['date_time'])



df.head()
df.plot(x = 'date_time', 

        y = 'price_usd', 

        figsize = (16, 8))



plt.xlabel('dates')

plt.ylabel('USD')

plt.title('Time series of room price by date of search');
df.head()
a = df.loc[df['srch_saturday_night_bool'] == 0, 'price_usd']

b = df.loc[df['srch_saturday_night_bool'] == 1, 'price_usd']



plt.figure(figsize = (16, 8))



plt.hist(a, bins = 80, 

         alpha = 0.3, 

         label = 'search w/o Sat night stay')



plt.hist(b, bins = 80, 

         alpha = 0.3, 

         label = 'search w/ Sat night stay')



plt.xlabel('Price')

plt.ylabel('Freq')

plt.legend()

plt.title('Sat night search')

plt.plot();
df['srch_saturday_night_bool'].value_counts()
print('Kurtosis: %f' % df['price_usd'].kurt())

print('Skewness: %f' % df['price_usd'].skew())
sns.distplot(df['price_usd'], 

                 hist = False, label = 'USD')



sns.distplot(df['srch_booking_window'], 

                  hist = False, label = 'booking window')



plt.xlabel('dist')

sns.despine()
sns.distplot(a, hist = False, rug = False)

sns.distplot(b, hist = False, rug = False)



sns.despine()
df = df.sort_values('date_time')

df['date_time_int'] = df.date_time.astype(np.int64)
# Determine optimal cluster num using elbow method 

data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

n_cluster = range(1, 20)



kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]

scores = [kmeans[i].score(data) for i in range(len(kmeans))]
# elbow curve 

fig, ax = plt.subplots(figsize = (16, 8))

ax.plot(n_cluster, scores, color = 'orange')



plt.xlabel('clusters num')

plt.ylabel('score')

plt.title('elbow curve')

plt.show();
# k means output 

X = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

X = X.reset_index(drop = True)



km = KMeans(n_clusters = 7)

km.fit(X)

km.predict(X)



labels = km.labels_



X.head()
fig = plt.figure(1, figsize = (7, 7))



ax = Axes3D(fig, rect = [0, 0, 0.95, 1], 

            elev = 48, azim = 134)



ax.scatter(X.iloc[:, 0], 

           X.iloc[:, 1], 

           X.iloc[:, 2],

           c = labels.astype(np.float), edgecolor = 'm')



ax.set_xlabel('USD')

ax.set_ylabel('srch_booking_window')

ax.set_zlabel('srch_saturday_night_bool')



plt.title('K Means', fontsize = 10);
import pylab as pl 
#Y = df[['price_usd']]

#X = df[['srch_booking_window']]



#Nc = range(1, 20)

#kmeans = [KMeans(n_clusters = i) for i in Nc]



#score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]



#plt.figure(figsize = (16, 8))

#pl.plot(Nc, score)

#pl.xlabel('cluster num')

#pl.ylabel('score')

#pl.title('elbow curve')

#pl.show();
#pca = PCA(n_components = 1).fit(Y)



#pca_d = pca.transform(Y)

#pca_c = pca.transform(X)



#kmeans = KMeans(n_clusters = 7)

#kmeansoutput = kmeans.fit(Y)



#pl.figure('7 cluster k-means')

#pl.figure(figsize = (16, 8))



#pl.scatter(pca_c[:, 0], 

#           pca_d[:, 0], 

#           c = kmeansoutput.labels_)



#pl.xlabel('booking window')

#pl.ylabel('USD')

#pl.title('7 cluster')

#pl.show();
data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]



X = data.values

X_std = StandardScaler().fit_transform(X)



# Calc eigenvec cor & eig_vals of covar matrix 

mean_vec = np.mean(X_std, axis = 0)



cov_mat = np.cov(X_std.T)



eig_vals, eig_vecs = np.linalg.eig(cov_mat)



# eig_val,eig_vecs tuple

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]



eig_pairs.sort(key = lambda x: x[0], reverse = True)
# Calc explained var from eig_vals 

total = sum(eig_vals)



# Individual explained var 

var_exp = [(i/total)*100 for i in sorted(eig_vals, reverse = True)]



# Cumulative explained var 

cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize = (16, 8))

plt.bar(range(len(var_exp)), var_exp, 

        alpha = 0.5, align = 'center', 

        label = 'individual explained var', 

        color = 'r'

       )



plt.step(range(len(cum_var_exp)), cum_var_exp,

         where = 'mid',

         label = 'cumulative explained var')



plt.xlabel('principal components')

plt.ylabel('explained var ratio')

plt.legend(loc = 'best')

plt.show();
data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]



# Standardize features

X_std = StandardScaler().fit_transform(X)

data = pd.DataFrame(X_std)



# Reduce components to 2 

pca = PCA(n_components = 2)

data = pca.fit_transform(data)



# Standardize 2 new features 

scaler = StandardScaler()

np_scaled = scaler.fit_transform(data)

data = pd.DataFrame(np_scaled)
kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]



df['cluster'] = kmeans[7].predict(data)

df.index = data.index



df['pc1'] = data[0]

df['pc2'] = data[1]

df['cluster'].value_counts()
def getDistanceByPoint(data, model):

    distance = pd.Series()

    for i in range(0,len(data)):

        Xa = np.array(data.loc[i])

        Xb = model.cluster_centers_[model.labels_[i]-1]

        distance.set_value(i, np.linalg.norm(Xa-Xb))

    return distance
outliers_fraction = 0.01



distance = getDistanceByPoint(data, kmeans[9])

outlier_num = int(outliers_fraction * len(distance))



threshold = distance.nlargest(outlier_num).min()



df['anomaly'] = (distance >= threshold).astype(int)
fig, ax = plt.subplots(figsize = (12, 6))



colors = {0:'blue', 1:'red'}



ax.scatter(df['pc1'], df['pc2'], 

           c = df['anomaly'].apply(lambda x: colors[x]))



plt.xlabel('pc1')

plt.ylabel('pc2')

plt.show();
df = df.sort_values('date_time')

df['date_time'] = df.date_time.astype(np.int64)



# object with anomalies

a = df.loc[df['anomaly'] == 1, 

           ['date_time_int', 'price_usd']]



a
fig, ax = plt.subplots(figsize = (10, 5))



ax.plot(df['date_time_int'], df['price_usd'], 

        color = 'orange', label = 'Normal')



ax.scatter(a['date_time_int'], a['price_usd'],

           color = 'red', label = 'Anomaly')



plt.xlabel('time')

plt.ylabel('USD')

plt.legend()

plt.show();

df['anomaly'].unique()
a = df.loc[df['anomaly'] == 0, 'price_usd']

b = df.loc[df['anomaly'] == 1, 'price_usd']



fig, axs = plt.subplots(figsize = (10, 5))

axs.hist([a, b], 

         bins = 50, stacked = True, 

         color = ['orange', 'red'])



plt.show();
df.anomaly.value_counts()
df.head()
data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]



scaler = StandardScaler()

np_scaled = scaler.fit_transform(data)



data = pd.DataFrame(np_scaled)



# Isolation forest 

outliers_fraction = 0.01

ifo = IsolationForest(contamination = outliers_fraction)



ifo.fit(data)



df['anomaly1'] = pd.Series(ifo.predict(data))



fig, ax = plt.subplots(figsize = (10, 5))



a = df.loc[df['anomaly1'] == -1, ['date_time_int', 'price_usd']]



ax.plot(df['date_time_int'], df['price_usd'], 

        color = 'orange', label = 'Normal')



ax.scatter(a['date_time_int'], a['price_usd'], 

           color = 'red', label = 'Anomaly')



plt.legend()

plt.show();
df['anomaly1'].unique()
a = df.loc[df['anomaly1'] == 1, 'price_usd']

b = df.loc[df['anomaly1'] == -1, 'price_usd']



fig, ax = plt.subplots(figsize = (10, 5))



ax.hist([a, b],

        bins = 50, stacked = True, 

        color = ['orange', 'red'] )



plt.show();
df['anomaly1'].value_counts()
data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

scaler = StandardScaler()

np_scaled = scaler.fit_transform(data)



data = pd.DataFrame(np_scaled)



# Train 



osvm = OneClassSVM(nu = outliers_fraction, 

                   kernel = 'rbf', 

                   gamma = 0.01)



osvm.fit(data)



df['anomaly2'] = pd.Series(osvm.predict(data))
fig, ax = plt.subplots(figsize = (10, 5))



a = df.loc[df['anomaly2'] == -1, 

           ['date_time_int', 'price_usd']]



ax.plot(df['date_time_int'], df['price_usd'], 

        color = 'orange', 

        label = 'Normal')



ax.scatter(a['date_time_int'], a['price_usd'], 

           color = 'red', 

           label = 'Anomaly')



plt.legend()

plt.show();
df.head()
a = df.loc[df['anomaly2'] == 1, 'price_usd']

b = df.loc[df['anomaly2'] == -1, 'price_usd']



fig, ax = plt.subplots(figsize = (10, 5))



ax.hist([a, b], bins = 50, 

        stacked = True, color = ['orange','red'])



plt.show();
df['anomaly2'].value_counts()
df_class0 = df.loc[df['srch_saturday_night_bool'] == 0, 'price_usd']

df_class1 = df.loc[df['srch_saturday_night_bool'] == 1, 'price_usd']



fig, axs = plt.subplots(1, 2)



df_class0.hist(ax = axs[0], bins = 50, color = 'orange')

df_class1.hist(ax = axs[1], bins = 50, color = 'red');
envelope = EllipticEnvelope(contamination = outliers_fraction)



x_train = df_class0.values.reshape(-1, 1)

envelope.fit(x_train)



df_class0 = pd.DataFrame(df_class0)

df_class0['deviation'] = envelope.decision_function(x_train)

df_class0['anomaly'] = envelope.predict(x_train)
envelope = EllipticEnvelope(contamination = outliers_fraction)



x_train = df_class1.values.reshape(-1, 1)

envelope.fit(x_train)



df_class1 = pd.DataFrame(df_class1)

df_class1['deviation'] = envelope.decision_function(x_train)

df_class1['anomaly'] = envelope.predict(x_train)
df_class = pd.concat([df_class0, df_class1])

df['anomaly3'] = df_class['anomaly']



fig, ax = plt.subplots(figsize = (10, 5))



a = df.loc[df['anomaly3'] == -1, 

           ('date_time_int', 'price_usd')]



ax.plot(df['date_time_int'], df['price_usd'], 

        color = 'orange')



ax.scatter(a['date_time_int'], a['price_usd'],

          color = 'red')



plt.show();
df['anomaly3'].value_counts()
a = df.loc[df['anomaly3'] == 1, 'price_usd']

b = df.loc[df['anomaly3'] == -1, 'price_usd']



fig, ax = plt.subplots(figsize = (10, 5))

ax.hist([a, b], 

        bins = 50, stacked = True, 

        color = ['orange', 'red'])



plt.show();