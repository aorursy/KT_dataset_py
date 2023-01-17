import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



sns.set_style('whitegrid')

%config InlineBackend.figure_format = 'retina'

%matplotlib inline



from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))



import Geohash as geo

import matplotlib.ticker as plticker

from datetime import timedelta



## for preprocessing and machine learning

from sklearn.cluster import KMeans, k_means

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

import sklearn.linear_model as linear_model

from sklearn.preprocessing import MinMaxScaler



## for Deep-learing:

import keras

from keras import models

from keras import layers

from keras.layers import Dense

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.layers import LSTM

from keras.layers import Dropout
df = pd.read_csv('../input/training.csv')

df.head()
df.info() #no null values
plt.figure(figsize=(10,6))

sns.distplot(df['demand'], label='demand')



#demand follows a lognormal distribution
fig, ax = plt.subplots(figsize=(20,6))

dd_ts = df.groupby('day')['demand'].sum().reset_index()

sns.lineplot(x='day', y='demand', data=dd_ts, ax=ax)

plt.show()

#day trend shows demand growing with cyclical effects
from datetime import timedelta

df['day_time'] = df['day'].astype('str') +':'+ df['timestamp']

df['day_time2'] = df['day_time'].apply(lambda x: x.split(':'))

df['day_time3'] = df['day_time2'].apply(lambda x: timedelta(days=int(x[0]),hours=int(x[1]),minutes=int(x[2])))

df['dum_time'] = pd.Timestamp(2019,1,1).normalize() + df['day_time3']

df.drop(['day_time','day_time2','day_time3'],axis=1,inplace=True) # drop irrelevant columns

df.head()
df['time'] = df['dum_time'].dt.time

df['hour'] = df['dum_time'].dt.hour

df['minute'] = df['dum_time'].dt.minute
fig, ax = plt.subplots(figsize=(20,10))

dd_ts = df.groupby('hour')['demand'].sum().reset_index()

sns.lineplot(x='hour', y='demand', data=dd_ts, ax=ax)

loc = plticker.MultipleLocator() # this locator puts ticks at regular intervals

ax.xaxis.set_major_locator(loc)

plt.show()

#time trend shows peak in the morning and trough at 1900-2000hrs
daycycle_dict1 = {}

for i in range(1,8):

    j = i

    while j <= 61:

        daycycle_dict1[j] = i

        j+=7

df['daycycle'] = df['day'].apply(lambda x: daycycle_dict1[x])
fig, ax = plt.subplots(figsize=(20,10))

dd_ts = df.groupby('daycycle')['demand'].sum().reset_index()

sns.lineplot(x='daycycle', y='demand', data=dd_ts, ax=ax)

plt.show()

#Each day of the week has different demand. The 5th & 6th days could be the weekends due to the change in demand behaviour.
df['geohash6'].value_counts().sort_values().head()

#there are locations with 1 or little datapoints
df.groupby('geohash6').agg(

    {

         'demand':"median",    # median demand for each location

         'geohash6': "count",  # no. of datapoints for each location

    }

).plot(x="geohash6", y="demand", kind="scatter")
df['latitude'] = df['geohash6'].apply(lambda x: geo.decode_exactly(x)[0])

df['longitude'] = df['geohash6'].apply(lambda x: geo.decode_exactly(x)[1])
fig, ax = plt.subplots(figsize=(20,15))

dd_loc = df.groupby(['latitude','longitude'])['demand'].sum().reset_index()

sns.scatterplot(x='longitude', y='latitude', size='demand', sizes=(40, 400), data=dd_loc, ax=ax)

plt.show()
df2 = df.groupby(['geohash6','latitude','longitude'])['demand'].agg(['sum','std']).fillna(0).reset_index() #treat those with nan standard deviation as 0

df2.head()
X = df2.drop('geohash6',axis=1)

Xs  = StandardScaler().fit_transform(X)

Xs  = pd.DataFrame(Xs , columns = X.columns.values)

Xs.head()
def opt_clusters(X, scaling=StandardScaler, k=11):

    #choosing clusters with elbow within cluster sum square errors and silhouette score

    inertia = []

    silh = []

    #standardizing required

    Xs = StandardScaler().fit_transform(X)

    Xs = pd.DataFrame(Xs, columns = X.columns.values)

    for i in range(1,k):

        model = KMeans(n_clusters=i, random_state=0).fit(Xs)

        predicted = model.labels_

        inertia.append(model.inertia_)#low inertia = low cluster sum square error. Low inertia -> Clusters are more compact.

        if i>1:

            silh.append(silhouette_score(Xs, predicted, metric='euclidean')) #High silhouette score = clusters are well separated. The score is based on how much closer data points are to their own clusters (intra-dist) than to the nearest neighbor cluster (inter-dist): (cohesion + separation).  

    plt.plot(np.arange(1, k, step=1), inertia)

    plt.title('Innertia vs clusters')

    plt.xlabel('No. of clusters')

    plt.ylabel('Within Clusters Sum-sq (WCSS)')

    plt.show()

    plt.scatter(np.arange(2, k, step=1), silh)

    plt.title('Sihouette vs clusters')

    plt.xlabel('No. of clusters')

    plt.ylabel('Silhouette score')

    plt.show()
opt_clusters(Xs, scaling=StandardScaler, k=11)
#getting prediction and centroids

#select 6 clusters based on silhouette and WCSS

kmeans = KMeans(n_clusters=6, random_state=0).fit(Xs)

predicted = kmeans.labels_

centroids = kmeans.cluster_centers_

Xs['predicted'] = predicted #or X['predicted'] = predicted
df2['cluster'] = Xs['predicted']
fig, ax = plt.subplots(figsize=(20,15))

sns.scatterplot(x='longitude', y='latitude', size='sum', hue='cluster', palette=sns.color_palette("Dark2", 6), sizes=(40, 400), data=df2, ax=ax)

plt.show()
#create a dictionary for these locations

cluster_dict = df2[['geohash6','cluster']].set_index('geohash6')['cluster'].to_dict()

df['cluster'] = df['geohash6'].apply(lambda x: cluster_dict[x])

loc_dict = {0:'clust0', 1:'clust1', 2:'clust2', 3:'clust3', 4:'clust4', 5:'clust5'}

df['cluster'] = df['cluster'].apply(lambda x: loc_dict[x])
df.head()
#Grouping and pivoting each cluster's demand

lstm_df = df.groupby(['dum_time','cluster','daycycle','hour','minute'])['demand'].sum().reset_index()

lstm_df.set_index('dum_time', inplace=True)



lstm_df2 = pd.pivot_table(lstm_df, values = 'demand', index=['dum_time','daycycle','hour','minute'], columns = 'cluster').reset_index()

lstm_df2.set_index('dum_time',inplace=True)
lstm_df2.head()
#Standardizing demand min max for better processing of neural nets

series = lstm_df2.drop(['daycycle','hour','minute'],axis=1)

scaler = MinMaxScaler(feature_range = (0,1))

scaled = scaler.fit_transform(series.values)

series_ss = pd.DataFrame(scaled)



series_ss.columns = list(series.columns)

series_ss.set_index(series.index, inplace=True)

series_ss.head()
#get dataset for each time series forecast t+1, t+2, t+3, t+4, t+5

tseries_dict = {}

for step in range(1,6):

    y = series_ss.copy()

    ts_prior = series_ss.shift(step)

    y.columns = [j + str(step) for j in list(series_ss.columns)]

    dummies = pd.get_dummies(lstm_df2[['daycycle','hour','minute']], columns = ['daycycle','hour','minute'],drop_first=True)

    tseries_dict[step] = pd.concat([ts_prior,dummies,y],axis=1).dropna()
###### function to split into train and test sets

def train_test_prep(data):

    values = data.values

    n_train_time = round(0.9*len(data))

    train = values[:n_train_time, :]

    test = values[n_train_time:, :]

    ##test = values[n_train_time:n_test_time, :]

    # split into input and outputs

    train_X, train_y = train[:, :-6], train[:, -6:]

    test_X, test_y = test[:, :-6], test[:, -6:]

    # reshape input to be 3D [samples, timesteps, features]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    train_X.shape, train_y.shape, test_X.shape, test_y.shape

    return train_X, train_y, test_X, test_y

# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].
# print(round(0.9*len(tseries_dict[1])))

# tseries_dict[1].iloc[5256,:]

# date where training set starts is 2019-02-25 22:00:00
#split into train/test X&y datasets for each step forecast

Xts_train = {}

Xts_test = {}

yts_train = {}

yts_test = {}

for step in range(1,6):

    Xts_train[step], yts_train[step], Xts_test[step], yts_test[step] = train_test_prep(tseries_dict[step])



tseries_dict[step].shape
#define training model

def model_train(train_X,train_y,test_X,test_y):

    model = Sequential()

    model.reset_states()

    model.add(LSTM(input_shape=(train_X.shape[1], train_X.shape[2]), output_dim=50, return_sequences = True))

    model.add(Dropout(0.5))

    model.add(LSTM(256))

    model.add(Dropout(0.5))

    model.add(Dense(6))

    model.compile(loss='mean_squared_error', optimizer='adam')    

    history = model.fit(train_X, train_y, epochs=20, batch_size=100, validation_data=(test_X, test_y), verbose=0, shuffle=False)        

    return model
#train and save models forecasting t+1,t+2,...,t+5

models_dict = {}



for step in range(1,6):

    model = model_train(Xts_train[step], yts_train[step], Xts_test[step], yts_test[step])

    model.save('timestep_'+str(i)+'.h5')

    models_dict[step] = model
#get error and forecast plot on test set

def model_predict(model,test_X,test_y):

    # make a prediction

    yhat = model.predict(test_X)

    # invert scaling for forecast

    yhat = yhat.reshape((len(test_y), 6))

    inv_yhat = scaler.inverse_transform(yhat)



    # invert scaling for actual

    test_y = test_y.reshape((len(test_y), 6))

    inv_y = scaler.inverse_transform(test_y)



    rmse_ls = []

    for i in range(0,6):

        rmse_ls.append(mean_squared_error(pd.DataFrame(inv_y)[i], pd.DataFrame(inv_yhat)[i]))



    mean_val = pd.DataFrame(inv_yhat).mean()

    error_df = pd.concat([pd.DataFrame(rmse_ls, columns=['rmse']),mean_val],axis=1)

    error_df['%_error'] = error_df['rmse']/error_df[0]*100

    error_df.columns = ['rmse','mean_val','%_error']

    print(error_df)

    

#     for i in range(0,6):

#         print(series.columns[i])

#         plt.plot(inv_y[:,i], marker='.', label="actual")

#         plt.plot(inv_yhat[:,i], 'r', label="prediction")

#         plt.legend(fontsize=10)

#         plt.show()

    

    fig, ax = plt.subplots(2, 3, figsize=(20,10))

    ax[0, 0].plot(inv_y[:,0], marker='.', label="actual")

    ax[0, 1].plot(inv_y[:,1], marker='.', label="actual")

    ax[0, 2].plot(inv_y[:,2], marker='.', label="actual")

    ax[1, 0].plot(inv_y[:,3], marker='.', label="actual")

    ax[1, 1].plot(inv_y[:,4], marker='.', label="actual")

    ax[1, 2].plot(inv_y[:,5], marker='.', label="actual")

    ax[0, 0].plot(inv_yhat[:,0], marker='.', label="prediction")

    ax[0, 1].plot(inv_yhat[:,1], marker='.', label="prediction")

    ax[0, 2].plot(inv_yhat[:,2], marker='.', label="prediction")

    ax[1, 0].plot(inv_yhat[:,3], marker='.', label="prediction")

    ax[1, 1].plot(inv_yhat[:,4], marker='.', label="prediction")

    ax[1, 2].plot(inv_yhat[:,5], marker='.', label="prediction")

    ax[0, 0].title.set_text('Cluster0')

    ax[0, 1].title.set_text('Cluster1')

    ax[0, 2].title.set_text('Cluster2')

    ax[1, 0].title.set_text('Cluster3')

    ax[1, 1].title.set_text('Cluster4')

    ax[1, 2].title.set_text('Cluster5')

    ax[0, 0].legend(fontsize=10,loc='upper right')

    plt.show()

    return inv_yhat
#save predicted cluster demand for each cluster for t+1,..t+5 into a data dictionary

y_pred = {}

for step in range(1,6):

    print('Forecast time ahead ', step)

    y_pred[step] = model_predict(models_dict[step],Xts_test[step], yts_test[step])



#error increase as timestep ahead to predict increases
#standardize lat and long for analysis later

from sklearn.preprocessing import MinMaxScaler

lat_long = MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(df[['latitude','longitude']])

lat_long = pd.DataFrame(lat_long)

lat_long.columns = ['latitude','longitude']



df['latitude'] = lat_long['latitude']

df['longitude'] = lat_long['longitude']
#get proportion of demand for each geolocation, relative to the cluster and time

sector_dd = df.groupby(['cluster','dum_time'])['demand'].sum().reset_index()

df_sect_dd = df.merge(sector_dd, left_on = ['cluster','dum_time'], right_on = ['cluster','dum_time'],how = 'inner',suffixes=['','_sect'])

df_sect_dd['prop_dd'] = df_sect_dd['demand']/df_sect_dd['demand_sect']

df_sect_dd['prop_dd'] = df_sect_dd['prop_dd'].fillna(0)

df_sect_dd.head()
#get required data for cross section analysis

df3 = df_sect_dd[['dum_time','geohash6','latitude','longitude','daycycle','cluster','hour','minute','demand','demand_sect','prop_dd']]

#training set will be prior to '2019-02-25 22:00:00', as per time series forecast

df3_train = df3[df3['dum_time'] < pd.Timestamp(2019,2,25,22,0)].drop(['dum_time'],axis=1)

df3_test = df3[df3['dum_time'] >= pd.Timestamp(2019,2,25,22,0)].drop(['dum_time'],axis=1)
#feature engineer for train set

#insert train set demand & prop_dd further statistics - mean,median,std,min,max



f = {'demand': ['median','std','mean','min','max'],'prop_dd': ['median','std','mean','min','max']}

df3_train2 = df3_train.groupby(['geohash6','cluster','daycycle','hour','minute']).agg(f).reset_index()

df3_train2.columns = ["".join(x) for x in df3_train2.columns.ravel()]

df3_train2 = df3_train2.fillna(0)
#Check consistency/fluctuations of location proportion dd for each day cycle's time interval

print('median prop_dd:', np.log(df3_train2['prop_ddstd'].median()))

sns.distplot(np.log(df3_train2[df3_train2['prop_ddstd']!=0]['prop_ddstd']))

plt.show()

#fluctuation of prop_dd has a lognormal distribution but a heavier left tail. Most geolocation has low fluctuations in proportion demand, consistency is present / low data available for these geolocations as well.. 
#as we are predicting prop_dd (before getting actual demand from multiplying cluster forecast), 

#remove demand from the data set and use prop_dd as target variable.

#merge geolocation historic features from training set to test set (features are acquired prior to test time)

df3_train_feat = df3_train.drop(['demand'],axis=1).merge(df3_train2, left_on = ['geohash6','cluster','daycycle','hour','minute'], 

                                                         right_on = ['geohash6','cluster','daycycle','hour','minute'],

                                                         how = 'inner',suffixes=['','_feat'])

df3_test_feat = df3_test.drop(['demand'],axis=1).merge(df3_train2, left_on = ['geohash6','cluster','daycycle','hour','minute'], 

                                                       right_on = ['geohash6','cluster','daycycle','hour','minute'],

                                                       how = 'inner',suffixes=['','_feat'])
df3_train_feat.head()
#train-test split

#dummy obj variables

#dont use geohash6 as dummy, giving too sparse matrix. use lat and longitude instead, under continuous variables

X_train = pd.get_dummies(df3_train_feat,columns=['cluster','daycycle','hour','minute'],

                         drop_first=True).drop(['prop_dd','geohash6'],axis=1).values

y_train = df3_train_feat['prop_dd'].values



X_test = pd.get_dummies(df3_test_feat,columns=['cluster','daycycle','hour','minute'],

                        drop_first=True).drop(['prop_dd','geohash6'],axis=1).values

y_test = df3_test_feat['prop_dd'].values
lm = linear_model.LinearRegression()

#fit model

model_lm = lm.fit(X_train, y_train)

predictions = model_lm.predict(X_test)
print('In-sample R-sq:',model_lm.score(X_train, y_train))

print('Out-sample R-sq:',model_lm.score(X_test, y_test))

print('MSE:', mean_squared_error(y_test, predictions))
resid = y_test - predictions

plt.scatter(resid, y_test)

#there are still unobserved linear relationship present, must further tease out in next iteration
#get dummy daterange for the forecasted clusters' demand

date_first = pd.Timestamp(2019,2,25,22,0)

date_last = df['dum_time'].max() 

test_dum_time = pd.date_range(date_first, date_last, freq='15min')
ts_ypred1 = pd.DataFrame(y_pred[1])

ts_ypred1['dum_time'] = test_dum_time

ts_ypred1.columns=['clust0','clust1','clust2','clust3','clust4','clust5','dum_time']

ts_ypred1 = pd.melt(ts_ypred1, id_vars=['dum_time'], value_vars=['clust0','clust1','clust2','clust3','clust4','clust5'])

ts_ypred1.columns=['dum_time','cluster','demand_sect']

ts_ypred1.head()
test_df = df3[df3['dum_time'] >= pd.Timestamp(2019,2,25,22,0)]

test_df.head()

test_df = test_df.merge(ts_ypred1, left_on = ['dum_time','cluster'], right_on = ['dum_time','cluster'],how = 'inner',suffixes=['','_pred'] )
test_df.head()
#merge geolocation historic features from training set (features acquired prior to test time)

test_df_feat = test_df.merge(df3_train2, left_on = ['geohash6','cluster','daycycle','hour','minute'], 

                             right_on = ['geohash6','cluster','daycycle','hour','minute'],

                             how = 'inner',suffixes=['','_feat'])

test_df_feat.head()
#get dummies into test set, remove irrelevant columns

test_df_feat2 = pd.get_dummies(test_df_feat,columns=['cluster','daycycle','hour','minute'],

                           drop_first=True).drop(['prop_dd','geohash6'],axis=1)



# use demand_sect_pred instead of demand_sect

X_test_cols = ['latitude', 'longitude', 'demand_sect_pred', 'demandmedian', 'demandstd',

       'demandmean', 'demandmin', 'demandmax', 'prop_ddmedian', 'prop_ddstd',

       'prop_ddmean', 'prop_ddmin', 'prop_ddmax', 'cluster_clust1',

       'cluster_clust2', 'cluster_clust3', 'cluster_clust4', 'cluster_clust5',

       'daycycle_2', 'daycycle_3', 'daycycle_4', 'daycycle_5', 'daycycle_6',

       'daycycle_7', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5',

       'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',

       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',

       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'minute_15',

       'minute_30', 'minute_45']

X_test_ts1 = test_df_feat2[X_test_cols].values

y_test_ts1 = test_df_feat2['demand'].values
pred_propdd_ts1 = model_lm.predict(X_test_ts1)
test_df_feat['pred_propdd_ts1'] = pd.DataFrame(pred_propdd_ts1)

test_df_feat['pred_demand_ts1'] = test_df_feat['demand_sect_pred'] * test_df_feat['pred_propdd_ts1']

#cap prediction with demand>1 to 1

test_df_feat['pred_demand_ts1'] = test_df_feat['pred_demand_ts1'].apply(lambda x: 1 if x>1 else x)

test_df_feat['resid_ts1'] = test_df_feat['demand'] - test_df_feat['pred_demand_ts1']
mean_val = test_df_feat['demand'].mean()

rmse = mean_squared_error(test_df_feat['pred_demand_ts1'], test_df_feat['demand'])

error_perc_ts1 = rmse/mean_val*100

print('Demand mean value:', mean_val)

print('RMSE:', rmse)

print('%_error:', error_perc_ts1)
plt.scatter(x=test_df_feat['pred_demand_ts1'], y=test_df_feat['demand'])

#prediction largely aligns with actual demand
plt.scatter(x=test_df_feat['resid_ts1'], y=test_df_feat['demand'])

#prediction largely aligns with actual demand