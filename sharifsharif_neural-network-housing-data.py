# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams

%matplotlib inline

rcParams['figure.figsize'] = 10,8

sns.set(style='whitegrid', palette='muted',

        rc={'figure.figsize': (15,10)})

import os

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder

from keras.wrappers.scikit_learn import KerasRegressor

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



from numpy.random import seed

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
housing = pd.read_csv("../input/california-housing-prices/housing.csv")

housing.head()
housing.dropna(inplace=True) # Drop all the null values 
housing['ocean_proximity'] = housing['ocean_proximity'].astype('category')

# convert to category codes

housing['ocean_proximity'] = housing['ocean_proximity'].cat.codes
# contData = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value']



# scaler = MinMaxScaler()

# for var in contData:

#     housing[var] = housing[var].astype('float64')

#     housing[var] = scaler.fit_transform(housing[var].values.reshape(-1, 1)) #data need reshaping in order to be used with fit_transform

scaler = MinMaxScaler()

housing = (housing-housing.min())/(housing.max()-housing.min())

housing.describe()
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)

display_all(housing.describe(include='all').T)
from sklearn.model_selection import train_test_split

set_X = housing.drop('median_house_value',axis= 1) # for the input we want all the coulmns but the median_house_value

set_Y = housing['median_house_value'] # for the output we want all the columns 
train_set_X , test_set_X, train_set_Y , test_set_Y = train_test_split(set_X,set_Y , test_size = 0.2 , random_state = 69)
train_set_X
def create_model(lyrs=[8,8], act='relu', opt='Adam', dr=0.0):

    

  

    model = Sequential()

    

    # create first hidden layer

    model.add(Dense(lyrs[0], input_dim = train_set_X.shape[1], activation=act))

    

    # create additional hidden layers

    for i in range(1,len(lyrs)):

        model.add(Dense(lyrs[i], activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1,activation='relu'))  # output layer activation='sigmoid')

    

    #Configuration of  the learning process

#     model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

    model.compile(optimizer='adam',loss='mse' , metrics = ['mae'])

    

    return model
model = create_model()

print(model.summary())
training = model.fit(train_set_X, train_set_Y, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

val_acc = np.mean(training.history['mae'])

print("\n%s: %.2f%%" % ('Mean squared error ', val_acc))



training.history['mae']

# Show model accuracy on graph for comparison

plt.plot(training.history['mae'])

plt.plot(training.history['val_mae'])

plt.title('Model Train/Validation Comparison')

plt.ylabel('mean_squared_error')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()



# Show model loss on graph for comparison

plt.plot(training.history['loss'])

plt.plot(training.history['val_loss'])

plt.title('Model Train/Validation Comparison')

plt.ylabel('Loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
model = KerasRegressor(build_fn=create_model, verbose=0)



# define the grid search parameters

batch_size = [16, 32, 64]

epochs = [25, 50,100]

param_grid = dict(batch_size=batch_size, epochs=epochs)



# search the grid

grid = GridSearchCV(estimator=model, 

                    param_grid=param_grid,

                    cv=3,

                    verbose=0)  # include n_jobs=-1 if you are using CPU



grid_result = grid.fit(train_set_X, train_set_Y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))