import sys

import pandas

import numpy

import sklearn

import keras



print('Python: {}'.format(sys.version))

print('Pandas: {}'.format(pandas.__version__))

print('Numpy: {}'.format(numpy.__version__))

print('Sklearn: {}'.format(sklearn.__version__))

print('Keras: {}'.format(keras.__version__))
import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np



# import the uci pima indians diabetes dataset

df = pd.read_csv('../input/pima_diabetes.csv')
df.info()
df.head()
# Describe the dataset

df.describe().T
df[df['Glucose'] == 0]
# Preprocess the data, mark zero values as NaN and drop

columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']



for col in columns:

    df[col].replace(0, np.NaN, inplace=True)

    

df.describe()
# Drop rows with missing values

df.dropna(inplace=True)



# summarize the number of rows and columns in df

df.describe()
# Convert dataframe to numpy array

dataset = df.values

print(dataset.shape)
# split into input (X) and an output (Y)

X = dataset[:,0:8]

Y = dataset[:, 8].astype(int)
print(X.shape)

print(Y.shape)

print(Y[:5])
# Normalize the data using sklearn StandardScaler

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler().fit(X)
print(scaler)
# Transform and display the training data

X_standardized = scaler.transform(X)



data = pd.DataFrame(X_standardized)

data.describe()
# import necessary sklearn and keras packages

from sklearn.model_selection import GridSearchCV, KFold

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import Adam
# Start defining the model

def create_model():

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = 0.01)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



model = create_model()

print(model.summary())
# Do a grid search for the optimal batch size and number of epochs

# import necessary packages

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import Adam



# Define a random seed

seed = 6

np.random.seed(seed)



# Start defining the model

def create_model():

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = 0.01)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



# create the model

model = KerasClassifier(build_fn = create_model, verbose = 1)



# define the grid search parameters

batch_size = [10, 20, 40]

epochs = [10, 50, 100]



# make a dictionary of the grid search parameters

param_grid = dict(batch_size=batch_size, epochs=epochs)



# build and fit the GridSearchCV

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)

grid_results = grid.fit(X_standardized, Y)



# summarize the results

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']

stds = grid_results.cv_results_['std_test_score']

params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
# Do a grid search for learning rate and dropout rate

# import necessary packages

from keras.layers import Dropout



# Define a random seed

seed = 6

np.random.seed(seed)



# Start defining the model

def create_model(learn_rate, dropout_rate):

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(dropout_rate))

    model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = learn_rate)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



# create the model

model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)



# define the grid search parameters

learn_rate = [0.001, 0.01, 0.1]

dropout_rate = [0.0, 0.1, 0.2]



# make a dictionary of the grid search parameters

param_grid = dict(learn_rate=learn_rate, dropout_rate=dropout_rate)



# build and fit the GridSearchCV

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)

grid_results = grid.fit(X_standardized, Y)



# summarize the results

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']

stds = grid_results.cv_results_['std_test_score']

params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
# Do a grid search to optimize kernel initialization and activation functions

# import necessary packages



# Define a random seed

seed = 6

np.random.seed(seed)



# Start defining the model

def create_model(activation, init):

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim = 8, kernel_initializer= init, activation= activation))

    model.add(Dense(4, input_dim = 8, kernel_initializer= init, activation= activation))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = 0.001)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



# create the model

model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)



# define the grid search parameters

activation = ['softmax', 'relu', 'tanh', 'linear']

init = ['uniform', 'normal', 'zero']



# make a dictionary of the grid search parameters

param_grid = dict(activation = activation, init = init)



# build and fit the GridSearchCV

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)

grid_results = grid.fit(X_standardized, Y)



# summarize the results

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']

stds = grid_results.cv_results_['std_test_score']

params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
# Do a grid search to find the optimal number of neurons in each hidden layer

# import necessary packages



# Define a random seed

seed = 6

np.random.seed(seed)



# Start defining the model

def create_model(neuron1, neuron2):

    # create model

    model = Sequential()

    model.add(Dense(neuron1, input_dim = 8, kernel_initializer= 'uniform', activation= 'linear'))

    model.add(Dense(neuron2, input_dim = neuron1, kernel_initializer= 'uniform', activation= 'linear'))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = 0.001)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



# create the model

model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)



# define the grid search parameters

neuron1 = [4, 8, 16]

neuron2 = [2, 4, 8]



# make a dictionary of the grid search parameters

param_grid = dict(neuron1 = neuron1, neuron2 = neuron2)



# build and fit the GridSearchCV

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), refit = True, verbose = 10)

grid_results = grid.fit(X_standardized, Y)



# summarize the results

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']

stds = grid_results.cv_results_['std_test_score']

params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
# generate predictions with optimal hyperparameters

y_pred = grid.predict(X_standardized)
print(y_pred.shape)
print(y_pred[:5])
# Generate a classification report

from sklearn.metrics import classification_report, accuracy_score



print(accuracy_score(Y, y_pred))

print(classification_report(Y, y_pred))