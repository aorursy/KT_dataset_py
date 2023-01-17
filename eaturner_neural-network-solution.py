# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



#import modules for kernel

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.regularizers import l2
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')

test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,

                    train[['dropoff_latitude', 'dropoff_longitude']].values,

                    test[['pickup_latitude', 'pickup_longitude']].values,

                    test[['dropoff_latitude', 'dropoff_longitude']].values))



pca = PCA().fit(coords)



sample_ind = np.random.permutation(len(coords))[:500000]

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])



def toDateTime( df ):

    

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday_name

    df['pickup_day'] = df['pickup_datetime'].dt.day

    df['pickup_month'] = df['pickup_datetime'].dt.month.astype('object')

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_minute'] = df['pickup_datetime'].dt.minute

    df['pickup_dt'] = (df['pickup_datetime'] - df['pickup_datetime'].min()).map(

        lambda x: x.total_seconds())

    

    df.drop('pickup_datetime', axis = 1, inplace = True)



    return df

#get radical distince

def haversine_np(lon1, lat1, lon2, lat2):

   

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km



#manhattan distance

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):



    a = haversine_np(lat1, lng1, lat1, lng2)

    b = haversine_np(lat1, lng1, lat2, lng1)

    return a + b



#bearing direction

def bearing_array(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))



#all distances

def locationFeatures( df ):

    #displacement of degrees

    df['up_town'] = np.sign( df['pickup_longitude'] - df['dropoff_longitude'] )

    df['est_side'] = np.sign( df['pickup_latitude'] - df['dropoff_latitude'] )

     

    #radical distances

    df['haversine_distance'] = haversine_np(

        df['pickup_longitude'], df['pickup_latitude'], 

        df['dropoff_longitude'], df['dropoff_latitude']

    )

    

    #log transform of the haversine distance

    df['log_haversine_distance'] = np.log1p(df['haversine_distance']) 

    

    #manhattan distances

    df['distance_dummy_manhattan'] = dummy_manhattan_distance(

        df['pickup_latitude'], df['pickup_longitude'],

        df['dropoff_latitude'], df['dropoff_longitude']

    )

    

    #log transform of the haversine distance

    df['log_distance_dummy_manhattan'] = np.log1p(df['distance_dummy_manhattan']) 

    

    #pca distances

    df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 0]

    df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 1]

    df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

    df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

       

    df.loc[:, 'pca_manhattan'] = ( np.abs(df['dropoff_pca1'] - df['pickup_pca1']) +

    np.abs(df['dropoff_pca0'] - df['pickup_pca0']) )

    

    df.loc[:, 'pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']]).astype('object')

    df.loc[:, 'dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']]).astype('object')

    

    df.drop(['pickup_longitude', 'dropoff_longitude'], axis = 1, inplace = True)

    df.drop(['pickup_latitude', 'dropoff_latitude'], axis = 1, inplace = True)

    

    return df



def featureCreate( df ):

    print ('Date time features')

    df = toDateTime( df )

    print ('Location Features')

    df = locationFeatures( df )

    

    return df
train = featureCreate( train )

test = featureCreate( test )



#log transform our trip duration

train['trip_duration'] = np.log1p(train['trip_duration'])
q1 = np.percentile(train['trip_duration'], 25)

q3 = np.percentile(train['trip_duration'], 75)



iqr = q3 - q1



train = train[ train['trip_duration'] <= q3 + 3.0*iqr]



train = train[ q1 - 3.0*iqr <= train['trip_duration']]
labels = train.pop('trip_duration')

train.drop(["id", "dropoff_datetime"], axis=1, inplace = True)



sub = pd.DataFrame( columns = ['id', 'trip_duration'])

sub['id'] = test.pop('id')
train = pd.get_dummies(train)

test = pd.get_dummies(test)
mmScale = MinMaxScaler()



#input shape

n = train.shape[1]



train = mmScale.fit_transform(train)
#deep learning

model = Sequential()

#Want to use an expotential linear unit instead of the usual relu

model.add( Dense( n, activation='relu', input_shape=(n,) ) )

model.add( Dense( int(0.5*n), activation='relu' ) )

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
print ('On to the next one')



model.fit(train, labels.values, epochs = 3)



print ('Finished')
train_pred = model.predict(train)

test_pred = model.predict( mmScale.transform(test) )
fig = plt.figure()

ax = fig.add_subplot(111)



ax.scatter(train_pred, labels, c='b', marker="s", label='pred')



plt.legend()



plt.show()
sub['trip_duration'] = np.expm1(test_pred)

sub.to_csv('submission.csv', index=False)