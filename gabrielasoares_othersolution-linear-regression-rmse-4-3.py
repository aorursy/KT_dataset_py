import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import sklearn.model_selection as ms

from sklearn import linear_model

import sklearn.metrics as sklm

import math

import warnings                  

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
test = pd.read_csv('../input/test_MV.csv')

test.describe()
train = pd.read_csv('../input/train_MV.csv')

train.describe()
train.drop('key', axis=1, inplace=True)

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train.dtypes
# Remove na

train = train.dropna()

train.shape
train = train[train['fare_amount'].between(left = 2.5, right = 500)]
train = train[train['passenger_count'].between(left = 1, right = 6)]
train = train.loc[train['pickup_latitude'].between(40.2, 41.4)]

train = train.loc[train['pickup_longitude'].between(-74.3, -72)]

train = train.loc[train['dropoff_latitude'].between(40.2, 41.4)]

train = train.loc[train['dropoff_longitude'].between(-74.3, -72)]
train.describe()
# add time information

train['year'] = train.pickup_datetime.apply(lambda t: t.year)

train['weekday'] = train.pickup_datetime.apply(lambda t: t.weekday())

train['hour'] = train.pickup_datetime.apply(lambda t: t.hour)
# This function is based on https://stackoverflow.com/questions/27928/

# calculate-distance-between-two-latitude-longitude-points-haversine-formula 

# return distance in miles

def distance(lat1, lon1, lat2, lon2):

    p = 0.017453292519943295 # Pi/180

    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2

    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
train['distance'] = distance(train.pickup_latitude, train.pickup_longitude, train.dropoff_latitude, 

                             train.dropoff_longitude)
# remove datapoints with distance <0.05 miles

idx = (train.distance >= 0.05)

print('Old size: %d' % len(train))

train = train[idx]

print('New size: %d' % len(train))
#nyc center coordinates

nyc = (-74.0063889, 40.7141667)

train['distance_to_center'] = distance(nyc[0], nyc[1], train.dropoff_latitude, train.dropoff_longitude)
train.columns
train['year'].unique(), train['weekday'].unique(), train['hour'].unique()
from sklearn import preprocessing



Features = train['weekday']

enc_weekday = preprocessing.LabelEncoder()

enc_weekday.fit(Features)

Features = enc_weekday.transform(Features)

ohe_weekday = preprocessing.OneHotEncoder()

encoded_weekday = ohe_weekday.fit(Features.reshape(-1,1))

Features = encoded_weekday.transform(Features.reshape(-1,1)).toarray()



temp = train['year']

enc_year = preprocessing.LabelEncoder()

enc_year.fit(temp)

temp = enc_year.transform(temp)

ohe_year = preprocessing.OneHotEncoder()

encoded_year = ohe_year.fit(temp.reshape(-1,1))

temp = encoded_year.transform(temp.reshape(-1,1)).toarray()

Features = np.concatenate([Features, temp], axis = 1)



temp = train['hour']

enc_hour=preprocessing.LabelEncoder()

enc_hour.fit(temp)

temp = enc_hour.transform(temp)

ohe_hour = preprocessing.OneHotEncoder()

encoded_hour = ohe_hour.fit(temp.reshape(-1,1))

temp = encoded_hour.transform(temp.reshape(-1,1)).toarray()

Features = np.concatenate([Features, temp], axis = 1)
Features.shape
Features = np.concatenate([Features, np.array(train[['passenger_count', 'distance', 'distance_to_center']])], axis = 1)

Labels = np.array(train['fare_amount'])
Features.shape, Labels.shape
Features[:2, 38:]
#Split data (train and validation)

X_train, X_valid, y_train, y_valid = train_test_split(Features, Labels, random_state=123, test_size=300000)
#rescale numeric features

scaler = preprocessing.StandardScaler().fit(X_train[:,38:])

X_train[:, 38:] = scaler.transform(X_train[:,38:])

X_valid[:, 38:] = scaler.transform(X_valid[:,38:])
#metrics

def print_metrics(y_true, y_predicted):

    ## First compute R^2 and the adjusted R^2

    r2 = sklm.r2_score(y_true, y_predicted)

    

    ## Print the usual metrics and the R^2 values

    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))

    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))

    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))

    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))

    print('R^2                    = ' + str(r2))
#models

lr = LinearRegression()

lr.fit(X_train, y_train)

y_score = lr.predict(X_valid)

print_metrics(y_valid,y_score)
lr.coef_
def plot_regularization(l, train_RMSE, test_RMSE, coefs, min_idx, title):   

    plt.plot(l, test_RMSE, color = 'red', label = 'Test RMSE')

    plt.plot(l, train_RMSE, label = 'Train RMSE')    

    plt.axvline(min_idx, color = 'black', linestyle = '--')

    plt.legend()

    plt.xlabel('Regularization parameter')

    plt.ylabel('Root Mean Square Error')

    plt.title(title)

    plt.show()

    

    plt.plot(l, coefs)

    plt.axvline(min_idx, color = 'black', linestyle = '--')

    plt.title('Model coefficient values \n vs. regularizaton parameter')

    plt.xlabel('Regularization parameter')

    plt.ylabel('Model coefficient value')

    plt.show()



def test_regularization_l2(x_train, y_train, x_test, y_test, l2):

    train_RMSE = []

    test_RMSE = []

    coefs = []

    for reg in l2:

        lin_mod = linear_model.Ridge(alpha = reg)

        lin_mod.fit(x_train, y_train)

        coefs.append(lin_mod.coef_)

        y_score_train = lin_mod.predict(x_train)

        train_RMSE.append(math.sqrt(sklm.mean_squared_error(y_train, y_score_train)))

        y_score = lin_mod.predict(x_test)

        test_RMSE.append(math.sqrt(sklm.mean_squared_error(y_test, y_score)))

    min_idx = np.argmin(test_RMSE)

    min_l2 = l2[min_idx]

    min_RMSE = test_RMSE[min_idx]

    

    title = 'Train and test root mean square error \n vs. regularization parameter'

    plot_regularization(l2, train_RMSE, test_RMSE, coefs, min_l2, title)

    return min_l2, min_RMSE

     

l2 = [x for x in range(1,101)]

out_l2 = test_regularization_l2(X_train, y_train, X_valid, y_valid, l2)

print(out_l2)
def test_regularization_l1(x_train, y_train, x_test, y_test, l1):

    train_RMSE = []

    test_RMSE = []

    coefs = []

    for reg in l1:

        lin_mod = linear_model.Lasso(alpha = reg)

        lin_mod.fit(x_train, y_train)

        coefs.append(lin_mod.coef_)

        y_score_train = lin_mod.predict(x_train)

        train_RMSE.append(math.sqrt(sklm.mean_squared_error(y_train, y_score_train)))

        y_score = lin_mod.predict(x_test)

        test_RMSE.append(math.sqrt(sklm.mean_squared_error(y_test, y_score)))

    min_idx = np.argmin(test_RMSE)

    min_l1 = l1[min_idx]

    min_RMSE = test_RMSE[min_idx]

    

    title = 'Train and test root mean square error \n vs. regularization parameter'

    plot_regularization(l1, train_RMSE, test_RMSE, coefs, min_l1, title)

    return min_l1, min_RMSE

    

l1 = [x/50 for x in range(1,101)]

out_l1 = test_regularization_l1(X_train, y_train, X_valid, y_valid, l1)

print(out_l1)
import sklearn.decomposition as skde

pca_mod = skde.PCA()

pca_comps = pca_mod.fit(X_train)

pca_comps
print(pca_comps.explained_variance_ratio_)
def plot_explained(mod):

    comps = mod.explained_variance_ratio_

    x = range(len(comps))

    x = [y + 1 for y in x]          

    plt.plot(x,comps)



plot_explained(pca_comps)
pca_mod_15 = skde.PCA(n_components = 15)

pca_mod_15.fit(X_train)

Comps = pca_mod_15.transform(X_train)

Comps.shape
lr_pca = LinearRegression()

lr_pca.fit(Comps, y_train)



X_valid_pca = pca_mod_15.transform(X_valid)

y_score_pca = lr_pca.predict(X_valid_pca)



print_metrics(y_valid,y_score_pca)
#PCA entire training dataset

import sklearn.decomposition as skde

pca_all = skde.PCA()

pca_comps_all = pca_all.fit(Features)

plot_explained(pca_comps_all)
pca_mod_5 = skde.PCA(n_components = 5)

pca_mod_5.fit(Features)

Features_5 = pca_mod_5.transform(Features) 

lr_pca_5 = LinearRegression()

lr_pca_5.fit(Features_5, Labels)

y_score_pca_5 = lr_pca_5.predict(Features_5)

print_metrics(Labels,y_score_pca_5)

Features_15 = pca_mod_15.transform(Features) 

lr_pca_15 = LinearRegression()

lr_pca_15.fit(Features_15, Labels)

y_score_pca_15 = lr_pca_15.predict(Features_15)

print_metrics(Labels,y_score_pca_15)

test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])

# calculate features on the test dataset

test['year'] = test.pickup_datetime.apply(lambda t: t.year)

test['weekday'] = test.pickup_datetime.apply(lambda t: t.weekday())

test['hour'] = test.pickup_datetime.apply(lambda t: t.hour)

test['distance']= distance(test.pickup_latitude, test.pickup_longitude, test.dropoff_latitude, test.dropoff_longitude)

test['distance_to_center'] = distance(nyc[0], nyc[1], test.dropoff_latitude, test.dropoff_longitude)





#transform features

Features_test = test['weekday']

Features_test=enc_weekday.transform(Features_test)

Features_test = encoded_weekday.transform(Features_test.reshape(-1,1)).toarray()



temp = test['year']

temp = enc_year.transform(temp)

temp = encoded_year.transform(temp.reshape(-1,1)).toarray()

Features_test = np.concatenate([Features_test, temp], axis = 1)



temp = test['hour']

temp = enc_hour.transform(temp)

temp = encoded_hour.transform(temp.reshape(-1,1)).toarray()

Features_test = np.concatenate([Features_test, temp], axis = 1)



#numeric features

Features_test = np.concatenate([Features_test, np.array(test[['passenger_count', 'distance', 'distance_to_center']])], axis = 1)

Features_test[:,38:]=scaler.transform(Features_test[:,38:])



#using PCA with 15 components

Features_test_15 = pca_mod_15.transform(Features_test) 

# fit the test dataset



#something is wrong with pca Features_test_15..

#Simple Linear Regression

#RMSE public: 4.26



lr.fit(X_train, y_train)

preds = lr.predict(Features_test)

preds
# save submission csv

sub = pd.DataFrame({'key': test.key, 'fare_amount': preds})

sub.to_csv('submission_lr_newfeatures.csv', index = False)
