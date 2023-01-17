# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load libraries

import pandas as pd

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import AdaBoostRegressor

from sklearn.impute import SimpleImputer

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential

from keras.layers import Dense

from keras import losses

from keras import optimizers

from scipy import stats

import numpy as np
# load data

database_name = "/kaggle/input/craigslist-carstrucks-data/craigslistVehicles.csv"

#form_names = ['price', 'year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color']

dataset = pd.read_csv(database_name, low_memory=False)

print("Load complete.")
with open(database_name, 'r') as db:

    for index, record in zip(range(5), db):

        print(record)
def try_cast_float(val):

    try:

        return float(val)

    except ValueError:

        return np.NaN



# prepare data for analysis

# first, change appropriate values to a numeric type

numeric_coltypes = ['price', 'year', 'odometer']

# TODO: maybe eliminate fuel, size, standard, and color?

cat_coltypes = ['manufacturer', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color']

drop_coltypes = ['url', 'city', 'city_url', 'make', 'VIN', 'image_url', 'desc', 'lat', 'long']



# first convert numeric columns to numerical values

for col in numeric_coltypes:

    try_cast_float(col)

    dataset[col] = dataset[col].apply(try_cast_float)



# convert useful string colums to category datatype

for col in cat_coltypes:

    dataset[col] = dataset[col].astype('category')



# trim unnecessary columns

for col in drop_coltypes:

    dataset.drop(columns=col, axis=1, inplace=True)

    

print(dataset.head(10))

print(dataset.dtypes)
# eliminate rows where the price is too low or too high

for i, price in zip(range(dataset.shape[0]), dataset['price']):

    if((price > 100000.0) or (price < 100.0)):

        dataset.drop(index=i, inplace=True)

    if(i % 10000 == 0):

        print("Iteration {}".format(i))

print(dataset.shape)
print(dataset.describe())
# box and whisker plots

dataset.plot(kind='box', subplots=True, layout=(1,3), sharex=False, sharey=False)

plt.show()
dataset.hist()

plt.show()
# scatter plot matrix

scatter_matrix(dataset)

plt.show()
# one-hot encode categorical types

ohe_dataset = pd.get_dummies(dataset)

filled_dataset = ohe_dataset.fillna(value=0)

imputed_dataset = SimpleImputer().fit_transform(filled_dataset)



print(filled_dataset.head(5))

print("~~~~~~~~~~")

print(imputed_dataset[0:5,0:5])
# prepare data for chucking in to a learning algorithm

X_features = imputed_dataset[10000:100000,1:]

Y_result = imputed_dataset[10000:100000,0]



# X_train,     X_valid,       Y_train,      Y_valid

feature_train, feature_valid, result_train, result_valid = train_test_split(X_features, Y_result, test_size=0.2, random_state=1)



print(X_features.shape)

print(Y_result.shape)
# scale and normalize the training data

scaler = StandardScaler()

feature_train = scaler.fit_transform(feature_train)

feature_valid = scaler.transform(feature_valid)



print(feature_train[0:5,0:10])

print("----------")

print(feature_valid[0:5,0:10])
# build a basic model and work with it

model = Sequential()

model.add(Dense(128,input_dim=103, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='relu'))



model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae','mse'])

history = model.fit(feature_train, result_train, epochs=10, batch_size=128, validation_data=(feature_valid, result_valid))
loss, mae, mse = model.evaluate(feature_train, result_train)

print("Loss: {}\nMean Absolute Error: {}\nMean Squared Error: {}".format(loss, mae, mse))
history.history.keys()
predictions = model.predict(feature_valid)

mapped_predictions = []

for prediction in predictions:

  mapped_predictions.append(np.argmax(prediction))

predictions = np.array(mapped_predictions)
plt.plot(history.history['mae'])

plt.plot(history.history['val_mae'])

plt.title('model mae')

plt.ylabel('mae')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.plot()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# WARNING: DEPRECATED, DO NOT RUN



# TODO: mean and standard deviations

# now working on the data

abreg = AdaBoostRegressor()

params = {

 'n_estimators': [50, 100],

 'learning_rate' : [0.01, 0.05, 0.1, 0.5],

 'loss' : ['linear', 'square', 'exponential']

 }

gridsearch = GridSearchCV(abreg, params, cv=5)

gridsearch.fit(feature_train, result_train)