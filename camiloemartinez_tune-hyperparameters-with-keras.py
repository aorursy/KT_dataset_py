import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.utils import np_utils



import os

print(os.listdir("../input"))
iris=pd.read_csv('../input/Iris.csv')

iris.head()
sns.pairplot(iris.iloc[:,1:6],hue='Species')
sns.countplot('Species',data=iris)
# Create feature and target arrays

X = iris.iloc[:,1:5]

y = iris.iloc[:,-1]

# Label encode Class (Species)

encoder = preprocessing.LabelEncoder()

encoder.fit(y)

encoded_y = encoder.transform(y)

# One Hot Encode

y_dummy = np_utils.to_categorical(encoded_y)

# Split into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y_dummy, test_size = 0.2, random_state=123, stratify=y)

print('X_train:', X_train.shape)

print('X_test:', X_test.shape)
# Imports

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.optimizers import SGD



# Building the model

model = Sequential()

model.add(Dense(8, input_dim=4, activation='sigmoid'))

model.add(Dense(6, activation='sigmoid'))

model.add(Dense(3, activation='sigmoid'))



# Compiling the model

model.compile(loss = 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()
# training the model

model.fit(X_train, y_train, epochs=10, batch_size=30, verbose=0)
# Evaluating the model on the training and testing set

score = model.evaluate(X_train, y_train)

print("\n Training Accuracy:", score[1])
# use scikit-learn to grid search the batch size and epochs

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
# function to create model

def create_model():

    model = Sequential()

    model.add(Dense(8, input_dim=4, activation='sigmoid'))

    model.add(Dense(6, activation='sigmoid'))

    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss = 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model



# create model

model = KerasClassifier(build_fn=create_model, verbose=0)



# define the parameters to search in grid search 

batch_size = [10, 50, 100]

epochs = [10, 50, 100]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# function to create model

def create_model(optimizer='adam'):

    model = Sequential()

    model.add(Dense(8, input_dim=4, activation='sigmoid'))

    model.add(Dense(6, activation='sigmoid'))

    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model



# create model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)



# define the parameters to search in grid search 

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# function to create model

def create_model(learn_rate=0.01, momentum=0):

    model = Sequential()

    model.add(Dense(8, input_dim=4, activation='sigmoid'))

    model.add(Dense(6, activation='sigmoid'))

    model.add(Dense(3, activation='sigmoid'))

    optimizer = SGD(lr=learn_rate, momentum=momentum)

    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model



# create model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)



# define the parameters to search in grid search 

learn_rate = [0.01, 0.1, 0.2]

momentum = [0.2, 0.6, 0.9]

param_grid = dict(learn_rate=learn_rate, momentum=momentum)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# function to create model

def create_model(activation='relu'):

    model = Sequential()

    model.add(Dense(8, input_dim=4, activation=activation))

    model.add(Dense(6, activation=activation))

    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# create model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)



# define the parameters to search in grid search 

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

param_grid = dict(activation=activation)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
from keras.layers import Dropout

from keras.constraints import maxnorm



# function to create model

def create_model(dropout_rate=0.0, weight_constraint=0):

    model = Sequential()

    model.add(Dense(8, input_dim=4, activation='tanh', kernel_constraint=maxnorm(weight_constraint)))

    model.add(Dropout(dropout_rate))

    model.add(Dense(6, activation='tanh'))

    model.add(Dropout(dropout_rate))

    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# create model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)



# define the parameters to search in grid search 

weight_constraint = [0, 1, 2]

dropout_rate = [0.0, 0.1, 0.2]

param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# function to create model

def create_model(neurons=1):

    model = Sequential()

    model.add(Dense(neurons, input_dim=4, activation='tanh', kernel_constraint=maxnorm(1)))

    model.add(Dropout(0.1))

    model.add(Dense(6, activation='tanh'))

    model.add(Dropout(0.1))

    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# create model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)



# define the parameters to search in grid search 

neurons = [1, 5, 10, 15, 20, 25, 30]

param_grid = dict(neurons=neurons)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))