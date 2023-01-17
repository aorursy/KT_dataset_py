!pip install regressors

import plotly_express as px

import numpy as np 

import pandas as pd 

import os

import statsmodels.formula.api as sm

import statsmodels.sandbox.tools.cross_val as cross_val

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model as lm

from regressors import stats

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut
data = pd.read_csv("../input/listings_detail.csv")
d = data.drop(columns=['listing_url','scrape_id','zipcode','weekly_price','monthly_price','last_scraped','host_neighbourhood','state','market','smart_location','country_code','country','square_feet','latitude','longitude','is_location_exact','first_review','last_review','name','summary','space','description','experiences_offered','neighborhood_overview','notes','transit','access','interaction','house_rules','thumbnail_url','medium_url','picture_url','xl_picture_url','host_url','host_name','host_location','host_about','host_response_time','host_response_rate','host_acceptance_rate','host_thumbnail_url','host_picture_url','host_listings_count','street','neighbourhood_cleansed','bed_type','calendar_updated','calendar_last_scraped','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','requires_license','license','jurisdiction_names','calculated_host_listings_count'])

#Removing $ sign & , from prices

#Converting to numeric values as necessary

d['price'] = d['price'].str.replace('$', '')

d['price'] = d['price'].str.replace(',', '')

d["price"] = pd.to_numeric(d["price"])

d['security_deposit'] = d['security_deposit'].str.replace('$', '')

d['security_deposit'] = d['security_deposit'].str.replace(',', '')

d["security_deposit"] = pd.to_numeric(d["security_deposit"])

d['cleaning_fee'] = d['cleaning_fee'].str.replace('$', '')

d['cleaning_fee'] = d['cleaning_fee'].str.replace(',', '')

d["cleaning_fee"] = pd.to_numeric(d["cleaning_fee"])

d['extra_people'] = d['extra_people'].str.replace('$', '')

d['extra_people'] = d['extra_people'].str.replace(',', '')

d["extra_people"] = pd.to_numeric(d["extra_people"])

d['host_since_year'] = pd.DatetimeIndex(d['host_since']).year

d["host_since_year"] = pd.to_numeric(d["host_since_year"])

#dropping more unncessary or messy columns (property type, amenities, neighborhood have too categorcial values to map; city is too messy)

d = d.drop(columns=['id','host_id','property_type','host_since','host_verifications','city','amenities','has_availability','neighbourhood' ] )

#mapping categorical values to numeric values\

#true and false values...

#host_is_superhost

d['host_is_superhost'] = d['host_is_superhost'].map({'t': 1, 'f': 0})

#host_has_profile_pic

d['host_has_profile_pic'] = d['host_has_profile_pic'].map({'t': 1, 'f': 0})

#host_identity_verified

d['host_identity_verified'] = d['host_identity_verified'].map({'t': 1, 'f': 0})

#instant_bookable

d['instant_bookable'] = d['instant_bookable'].map({'t': 1, 'f': 0})

# is_business_travel_ready

d['is_business_travel_ready'] = d['is_business_travel_ready'].map({'t': 1, 'f': 0})

# require_guest_profile_picture

d['require_guest_profile_picture'] = d['require_guest_profile_picture'].map({'t': 1, 'f': 0})

# require_guest_phone_verification

d['require_guest_phone_verification'] = d['require_guest_phone_verification'].map({'t': 1, 'f': 0})

# other categories...

# cancellation_policy

d['cancellation_policy'] = d['cancellation_policy'].map({'super_strict_30': 3,'strict': 2,'moderate': 1, 'flexible': 0})

#room_type

d['room_type'] = d['room_type'].map({'Entire home/apt':2,'Private room': 1, 'Shared room': 0})

#neighbourhood_group_cleansed

d['neighbourhood_group_cleansed'] = d['neighbourhood_group_cleansed'].map({'Manhattan':4,'Brooklyn': 3, 'Queens': 2,'Bronx': 1, 'Staten Island': 0})

d=d.astype(float)

d=d.dropna()

#reset index

d = d.reset_index(drop=True)

d=d[[c for c in d if c not in ['price']] 

       + ['price']]

d.head()
d.shape
# Splitting



data = d.values

train_data = data[:,0:30]

train_price = data[:,29]

test_data = data[0:2000,0:30]

test_price = data[0:2000,29]

# print(train_price)
#Training Data Exploration

print("Dimensions of the training set: ", train_data.shape)

print("No. of labels in the training set: ",len(train_price))
#Test Data Exploration

print("Dimensions of the test set: ", test_data.shape)

print("No. of labels in the test set: ",len(test_price))
import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers.core import Dense, Activation



from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



# TensorFlow

import tensorflow as tf

from tensorflow import keras



print(tf.__version__)
X = data[0:100:,0:30]

Y = data[0:100:,29]

# define base model

def baseline_model():

	# create model

	model = Sequential()

	model.add(Dense(30, input_dim=30, kernel_initializer='normal', activation='relu'))

	model.add(Dense(1, kernel_initializer='normal'))

	# Compile model

	model.compile(loss='mean_squared_error', optimizer='adam')

	return model
# fix random seed for reproducibility

seed = 7

np.random.seed(seed)

# evaluate model with standardized dataset

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=5, random_state=seed)

results = cross_val_score(estimator, X, Y, cv=kfold)

print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate model with standardized dataset

np.random.seed(seed)

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold(n_splits=5, random_state=seed)

results = cross_val_score(pipeline, X, Y, cv=kfold)

print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# #import libraries

# import progressbar as pb



# #initialize widgets

# widgets = ['Time for loop of 100 iterations: ', pb.Percentage(), ' ', 

#             pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]

# #initialize timer

# timer = pb.ProgressBar(widgets=widgets, maxval=100).start()



# #for loop example

# for i in range(0,100):

#     results = cross_val_score(estimator, X, Y, cv=kfold)

#     print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#     timer.update(i)

# #finish

# timer.finish()
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from tensorflow.python.keras import optimizers
x = data[:,0:29]

y = data[:,29]



y=np.reshape(y, (-1,1))

scaler = MinMaxScaler()

print(scaler.fit(x))

print(scaler.fit(y))

xscale=scaler.transform(x)

yscale=scaler.transform(y)

print('\n',x.shape)
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)
del model
model = Sequential()

model.add(Dense(29,input_dim=29,kernel_initializer='uniform', activation='relu'))

model.add(Dense(29,activation='relu'))

model.add(Dense(58,activation='relu'))

model.add(Dense(290,activation='relu'))

model.add(Dense(580,activation='relu'))

model.add(Dense(290,activation='relu'))

model.add(Dense(58,activation='relu'))

model.add(Dense(29,activation='relu'))

model.add(Dense(1,activation='linear'))

model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=50,  verbose=1, validation_split=0.2)
print(history.history.keys())

# "Loss"

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model loss')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
print(model.summary())
PriceRec = model.predict(x)

for i in range(0,100):

    print(PriceRec[i].astype(int), '\t', y[i])

# print("R - Squared value:\n",stats.adj_r2_score(res, x_train, y_train)) 

print("RMSE:\n", np.sqrt(metrics.mean_squared_error(y, PriceRec)))