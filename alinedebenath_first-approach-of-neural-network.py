import warnings 

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.datasets import load_digits

from sklearn.ensemble import RandomForestClassifier

import random

from xgboost import XGBClassifier

from time import time

from keras.utils import np_utils, to_categorical

from keras.layers.core import Dense, Activation, Dropout

from keras.models import Sequential

from keras.wrappers.scikit_learn import KerasClassifier

from keras.constraints import maxnorm

from keras.callbacks import ReduceLROnPlateau
data = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

data.head()
data.isnull().any().sum()
train_data=(data.drop(columns='label')).values

labeled_data_num=(data.label).values

labeled_data=to_categorical(labeled_data_num)

train_data0 = train_data.reshape(train_data.shape[0], 28, 28)

plt.figure(figsize=(13,13))

for i in range(5, 9):

    plt.subplot(450 + (i+1))

    plt.imshow(train_data0[i], cmap=plt.get_cmap('gray'))

    plt.title(labeled_data_num[i])

    plt.axis('off')

plt.show
data.label.value_counts().sort_index().plot(kind='bar')
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='label'), data.label, test_size=0.2, random_state=2)
rfc  = RandomForestClassifier(n_estimators = 300,criterion= 'gini', max_depth= 5, max_features=9)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print('Accuracy of Random Forest model:', accuracy_score(y_test, rfc_pred))
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='label'), data.label, test_size=0.2, random_state=2)
xgb=XGBClassifier(objective='multi:softmax', num_class=10, 

        n_jobs=-1,booster="gbtree",tree_method = "hist",

        grow_policy = "depthwise")

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print('Accuracy of XGboost model:', accuracy_score(y_test, xgb_pred))
#Import and split data

X_train = (data.drop(columns='label').values).astype('float32')

#Preprocessing : converting labels into categorical

y_train = np_utils.to_categorical(data.label.values.astype('int32'))

#Preprocessing : normalization

scale = np.max(X_train)

X_train /= scale

mean = np.std(X_train)

X_train -= mean



input_dim = X_train.shape[1]

nb_classes = y_train.shape[1]
def create_model():

	# create model

	model = Sequential()

	model.add(Dense(128, input_dim=input_dim,kernel_initializer='he_uniform',kernel_constraint=maxnorm(1)))

	model.add(Activation('softplus'))

	model.add(Dropout(0.1))

	model.add(Dense(128, kernel_initializer='he_normal',kernel_constraint=maxnorm(2)))

	model.add(Activation('relu'))

	model.add(Dropout(0.1))

	model.add(Dense(nb_classes))

	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

	return model

model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters

batch_size = [20, 50, 100]

epochs = [20, 50, 80, 100]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_train, y_train)

# summarize results

print("Best batch size and epochs: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
#Grid search for optimizer

def create_model(optimizer='adam'):

	# create model

	model = Sequential()

	model.add(Dense(128, input_dim=input_dim,kernel_initializer='he_uniform',

                    kernel_constraint=maxnorm(2)))

	model.add(Activation('softplus'))

	model.add(Dropout(0.1))

	model.add(Dense(128,kernel_initializer='he_normal',kernel_constraint=maxnorm(1)))

	model.add(Activation('relu'))

	model.add(Dropout(0.1))

	model.add(Dense(nb_classes))

	model.add(Activation('softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=100, verbose=0)

# define the grid search parameters

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best optimizer: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
def create_model(activation='relu'):

	# create model

	model = Sequential()

	model.add(Dense(128, input_dim=input_dim,kernel_initializer='he_uniform',

                kernel_constraint=maxnorm(2), activation=activation))

	model.add(Dropout(0.1))

	model.add(Dense(128,kernel_initializer='lecun_uniform',kernel_constraint=maxnorm(1)))

	model.add(Activation('relu'))

	model.add(Dropout(0.1))

	model.add(Dense(nb_classes))

	model.add(Activation('softplus'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

	return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=100, verbose=0)

# define the grid search parameters

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

param_grid = dict(activation=activation)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best activation 1: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
def create_model(activation='relu'):

	# create model

	model = Sequential()

	model.add(Dense(128, input_dim=input_dim,kernel_initializer='he_uniform',

                kernel_constraint=maxnorm(2)))

	model.add(Activation('softplus'))

	model.add(Dropout(0.1))

	model.add(Dense(128,kernel_initializer='he_normal',kernel_constraint=maxnorm(1),

                    activation=activation))

	model.add(Dropout(0.1))

	model.add(Dense(nb_classes))

	model.add(Activation('softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

	return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=100, verbose=0)

# define the grid search parameters

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

param_grid = dict(activation=activation)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best activation 2: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
def create_model(activation='relu'):

	# create model

	model = Sequential()

	model.add(Dense(128, input_dim=input_dim,kernel_initializer='he_uniform',

                kernel_constraint=maxnorm(2)))

	model.add(Activation('softplus'))

	model.add(Dropout(0.1))

	model.add(Dense(128,kernel_initializer='lecun_uniform',kernel_constraint=maxnorm(1)))

	model.add(Activation('relu'))

	model.add(Dropout(0.1))

	model.add(Dense(nb_classes,activation=activation))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

	return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=100, verbose=0)

# define the grid search parameters

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

param_grid = dict(activation=activation)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best activation 3: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
def create_model(init_mode='uniform'):

	# create model

	model = Sequential()

	model.add(Dense(128, input_dim=input_dim,kernel_initializer=init_mode,

                kernel_constraint=maxnorm(2)))

	model.add(Activation('softplus'))

	model.add(Dropout(0.1))

	model.add(Dense(128,kernel_initializer='he_normal',kernel_constraint=maxnorm(1)))

	model.add(Activation('relu'))

	model.add(Dropout(0.1))

	model.add(Dense(nb_classes))

	model.add(Activation('softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

	return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=100, verbose=0)

# define the grid search parameters

init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

param_grid = dict(init_mode=init_mode)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best init mode: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
def create_model(init_mode='uniform'):

	# create model

	model = Sequential()

	model.add(Dense(128, input_dim=input_dim,kernel_initializer='he_uniform',

                kernel_constraint=maxnorm(2)))

	model.add(Activation('softplus'))

	model.add(Dropout(0.1))

	model.add(Dense(128,kernel_initializer=init_mode,kernel_constraint=maxnorm(1)))

	model.add(Activation('relu'))

	model.add(Dropout(0.1))

	model.add(Dense(nb_classes))

	model.add(Activation('softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

	return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=100, verbose=0)

# define the grid search parameters

init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

param_grid = dict(init_mode=init_mode)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_train, y_train)



# summarize results

print("Best init mode: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
#Import and split data

X_train = (data.drop(columns='label').values).astype('float32')

X_test = (pd.read_csv('../input/digit-recognizer/test.csv').values).astype('float32')

#Preprocessing : converting labels into categorical

y_train = np_utils.to_categorical(data.label.values.astype('int32'))

#Preprocessing : normalization

scale = np.max(X_train)

X_train /= scale

X_test /= scale

mean = np.std(X_train)

X_train -= mean

X_test -= mean



input_dim = X_train.shape[1]

nb_classes = y_train.shape[1]

#Model

input_dim = X_train.shape[1]

nb_classes = y_train.shape[1]

model = Sequential()

model.add(Dense(128, input_dim=input_dim,kernel_initializer='he_uniform',kernel_constraint=maxnorm(1)))

model.add(Activation('softplus'))

model.add(Dropout(0.1))

model.add(Dense(128, kernel_initializer='he_normal',kernel_constraint=maxnorm(2)))

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

print("Training...")

model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.1, verbose=2)
print("Generating test predictions...")

preds = model.predict_classes(X_test, verbose=0)



def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)



write_preds(preds, "keras-nlp.csv")

print("Predictions available")