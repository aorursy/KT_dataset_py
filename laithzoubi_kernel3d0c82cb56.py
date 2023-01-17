import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams

%matplotlib inline

rcParams['figure.figsize'] = 10,8

sns.set(style='whitegrid', palette='muted',

        rc={'figure.figsize': (15,10)})

import os

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# print(os.listdir("../input"))
tajrobah=pd.read_csv('../input/california-housing-prices/housing.csv' )

tajrobah
tajrobah
tajrobah.dropna(inplace=True) 

# convert to cateogry dtype

tajrobah['ocean_proximity'] = tajrobah['ocean_proximity'].astype('category')

# convert to category codes

tajrobah['ocean_proximity'] = tajrobah['ocean_proximity'].cat.codes

tajrobah
continuous = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population','households', 'median_income', 'median_house_value']



scaler = MinMaxScaler()

for var in continuous:

    tajrobah[var] = tajrobah[var].astype('float64')

    tajrobah[var] = scaler.fit_transform(tajrobah[var].values.reshape(-1, 1))
# display_all(tajrobah.describe(include='all').T)
X_train = tajrobah[pd.notnull(tajrobah['median_house_value'])].drop(['median_house_value'], axis=1)

y_train = tajrobah[pd.notnull(tajrobah['median_house_value'])]['median_house_value']
def create_model(lyrs=[10,10,20], act='relu', opt='Adam', dr=0.0):

    

  

    model = Sequential()

    

    # create first hidden layer

    model.add(Dense(lyrs[0], input_dim= X_train.shape[1], activation=act))

    

    # create additional hidden layers

    for i in range(1,len(lyrs)):

        model.add(Dense(lyrs[i], activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1, activation='relu'))  # output layer

    

    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    

    return model
model = create_model()

print(model.summary())
training = model.fit(X_train , y_train , epochs=100, batch_size=32, validation_split=0.2, verbose=0)

val_acc = np.mean(training.history['accuracy'])

print("\n%s: %.2f%%" % ('Accuracy', val_acc*100))
plt.plot(training.history['accuracy'])

plt.plot(training.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
model = KerasClassifier(build_fn=create_model, verbose=0)



# define the grid search parameters

batch_size = [16, 32, 64]

epochs = [50, 100]

param_grid = dict(batch_size=batch_size, epochs=epochs)



# search the grid

grid = GridSearchCV(estimator=model, 

                    param_grid=param_grid,

                    cv=3,

                    verbose=2)  # include n_jobs=-1 if you are using CPU



grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))