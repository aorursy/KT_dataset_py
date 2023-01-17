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
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import time

import tensorflow as tf 

from tensorflow import keras

from sklearn import tree

from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC, SVC

import xgboost

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.shape, test.shape
train.head(), test.head()
train.info(), train.describe(), test.info(), test.describe()
train.isnull().sum(), test.isnull().sum()
train_reduced = train[:5000]

train_val = train[37000:]
X = train_reduced.drop('label', axis=1)

y = train_reduced['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
train['label'].value_counts()/len(train), train_reduced['label'].value_counts()/len(train_reduced), y_train.value_counts()/len(y_train), y_test.value_counts()/len(y_test)
random_state=42

models = [tree.DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state), SGDClassifier(random_state=random_state), 

          LinearSVC(random_state=random_state, max_iter=10000), SVC(random_state=random_state, max_iter=10000),

            xgboost.XGBClassifier(random_state=random_state), AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1), 

          ExtraTreesClassifier(random_state=random_state), GradientBoostingClassifier(random_state=random_state)]

columns = ['Name', 'Score']

models_compare = pd.DataFrame(columns=columns)

i=0

for model in models:

    start_time = time.time()

    clf = model

    clf.fit(X_train, y_train)

    models_compare.loc[i, 'Name'] = clf.__class__.__name__

    models_compare.loc[i, 'Score'] = clf.score(X_test, y_test)

    models_compare.loc[i, 'Execution time'] = time.time()- start_time

    i+=1
models_compare.sort_values(by='Score', ascending=False)
g = sns.catplot(x="Name", y="Score", kind='point', aspect=4, markers="o", linestyles= "--", data=models_compare)
models = [tree.DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state), SGDClassifier(random_state=random_state), 

          LinearSVC(random_state=random_state, max_iter=10000), SVC(random_state=random_state, max_iter=10000),

            xgboost.XGBClassifier(random_state=random_state), AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1), 

          ExtraTreesClassifier(random_state=random_state), GradientBoostingClassifier(random_state=random_state)]

columns = ['Name', 'Score']

models_compare_cv = pd.DataFrame(columns=columns)

i=0

for model in models: 

    start_time = time.time()

    clf = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)

    models_compare_cv.loc[i, 'Name'] = model.__class__.__name__

    models_compare_cv.loc[i, 'Score'] = clf.mean()

    models_compare_cv.loc[i, 'Execution time'] = time.time()- start_time

    i+=1

models_compare_cv.sort_values(by='Score', ascending=False)
g = sns.catplot(x="Name", y="Score", kind='point', aspect=4, markers="o", linestyles= "--", data=models_compare_cv)
X_net = train_reduced.drop('label', axis=1)

y_net = train_reduced['label']
X_val = train_val.drop('label', axis=1)

y_val = train_val['label']
X_net = X_net.astype('float32')

X_net /= 255.0

X_val = X_val.astype('float32')

X_val /= 255.0
X_net = X_net.values.reshape(-1,28,28,1)

X_val = X_val.values.reshape(-1,28,28,1)
X_train_net, X_test_net, y_train_net, y_test_net = train_test_split(X_net, y_net, test_size=0.25, random_state=42)  
model_nn = keras.models.Sequential()

model_nn.add(keras.layers.Flatten(input_shape=[28,28,1]))

model_nn.add(keras.layers.Dropout(rate=0.2))

model_nn.add(keras.layers.Dense(300, activation='selu', kernel_initializer='lecun_normal'))

model_nn.add(keras.layers.Dropout(rate=0.2))

model_nn.add(keras.layers.Dense(300, activation='selu', kernel_initializer='lecun_normal'))

model_nn.add(keras.layers.Dropout(rate=0.2))

model_nn.add(keras.layers.Dense(300, activation='selu', kernel_initializer='lecun_normal'))

model_nn.add(keras.layers.Dropout(rate=0.2))

model_nn.add(keras.layers.Dense(10, activation='softmax'))
model_nn.summary()
K = keras.backend

class OneCycleScheduler(keras.callbacks.Callback):

    def __init__(self, iterations, max_rate, start_rate=None,

                 last_iterations=None, last_rate=None):

        self.iterations = iterations

        self.max_rate = max_rate

        self.start_rate = start_rate or max_rate / 10

        self.last_iterations = last_iterations or iterations // 10 + 1

        self.half_iteration = (iterations - self.last_iterations) // 2

        self.last_rate = last_rate or self.start_rate / 1000

        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):

        return ((rate2 - rate1) * (self.iteration - iter1)

                / (iter2 - iter1) + rate1)

    def on_batch_begin(self, batch, logs):

        if self.iteration < self.half_iteration:

            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)

        elif self.iteration < 2 * self.half_iteration:

            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,

                                     self.max_rate, self.start_rate)

        else:

            rate = self._interpolate(2 * self.half_iteration, self.iterations,

                                     self.start_rate, self.last_rate)

            rate = max(rate, self.last_rate)

        self.iteration += 1

        K.set_value(self.model.optimizer.lr, rate)
batch_size = 128

n_epochs = 25

onecycle = OneCycleScheduler(len(X_train_net) // batch_size * n_epochs, max_rate=0.05)

model_nn.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), metrics=['accuracy'])

history = model_nn.fit(X_train_net, y_train_net, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[onecycle])
pd.DataFrame(history.history).plot(figsize=(8,5))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()
model_nn.evaluate(X_test_net, y_test_net)
def build_model(n_hidden=1, n_neurons =30, learning_rate=3e-3, dropout_rate=0.2):

    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[28,28,1]))

    for layer in range(n_hidden):

        model.add(keras.layers.Dropout(rate=dropout_rate))

        model.add(keras.layers.Dense(n_neurons, activation='selu', kernel_initializer='lecun_normal'))

    model.add(keras.layers.Dense(10, activation='softmax'))

    optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
keras_class = keras.wrappers.scikit_learn.KerasClassifier(build_model)
param_distribs = {

    'n_hidden' : [3, 4, 5],

    'n_neurons' : [200, 250, 300],

    'learning_rate' : [0.01, 0.1, 1],

    'dropout_rate' : [0.2, 0.3, 0.4],

}
from sklearn.model_selection import RandomizedSearchCV

batch_size = 128

n_epochs = 25

onecycle = OneCycleScheduler(len(X_train_net) // batch_size * n_epochs, max_rate=0.05)

start_time = time.time()

rnd_search_cv = RandomizedSearchCV(keras_class, param_distribs, n_iter=30, cv=3)

rnd_search_cv.fit(X_train_net, y_train_net, verbose=0, epochs=30, validation_data=(X_val, y_val), callbacks=[onecycle])
print('Execution time:',time.time()- start_time)
rnd_search_cv.best_params_, rnd_search_cv.best_score_
model = rnd_search_cv.best_estimator_.model

model.evaluate(X_test_net, y_test_net)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

space = {'choice' : hp.choice('num_layers', [{'layers':'two', },

                    {'layers':'three',

                    'units3': hp.uniform('units3', 64,1024),}

                    ]),

            'units1': hp.uniform('units1', 64,1024),

            'units2': hp.uniform('units2', 64,1024),

            'batch_size' : hp.uniform('batch_size', 28,128),

            'learning_rate' : hp.choice('learning_rate', [0.0001, 0.001, 0.01, 0.1, 1])

        }
def model (params):

    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[28,28,1]))

    model.add(keras.layers.Dense(params['units1'], activation='selu', kernel_initializer='lecun_normal'))

    model.add(keras.layers.Dense(params['units2'], activation='selu', kernel_initializer='lecun_normal'))

    if params['choice']['layers'] == 'three':

        model.add(keras.layers.Dense(params['choice']['units3'], activation='selu', kernel_initializer='lecun_normal'))

    model.add(keras.layers.Dense(10, activation='softmax'))

    optimizer=keras.optimizers.SGD(learning_rate=params['learning_rate'], momentum=0.9)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train_net, y_train_net, verbose=0, batch_size=32,epochs=30, validation_data=(X_val, y_val), callbacks=[onecycle])

    test_score, test_acc = model.evaluate(X_test_net, y_test_net, verbose = 0)

    return{'loss': test_acc, 'score': test_score, 'status': STATUS_OK}
best = fmin(model, space , algo=tpe.suggest, max_evals=50, trials=Trials())
best
batch_size = 32

n_epochs = 50

model_cnn = keras.models.Sequential([

    keras.layers.Conv2D(64, 7, activation='relu', padding='same', input_shape=[28, 28, 1]),

    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(128,3,activation='relu', padding='same'),

    keras.layers.Conv2D(128,3,activation='relu', padding='same'),

    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(256,3,activation='relu', padding='same'),

    keras.layers.Conv2D(256,3,activation='relu', padding='same'),

    keras.layers.MaxPooling2D(2),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(10, activation='softmax')

    ])

model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

history = model_cnn.fit(X_train_net, y_train_net, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val))
pd.DataFrame(history.history).plot(figsize=(8,5))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()
model_cnn.evaluate(X_test_net, y_test_net)
class ResidualUnit(keras.layers.Layer):

    def __init__(self, filters, strides=1, activation='relu', **kwargs):

        super().__init__(**kwargs)

        self.activation = keras.activations.get(activation)

        self.main_layers = [

            keras.layers.Conv2D(filters, 3, strides = strides, padding='same', use_bias=False),

            keras.layers.BatchNormalization(),

            self.activation,

            keras.layers.Conv2D(filters,3 , strides=1, padding='same', use_bias=False),

            keras.layers.BatchNormalization()]

        self.skip_layers = []

        if strides > 1:

            self.skip_layers = [

                keras.layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),

                keras.layers.BatchNormalization()]

def call(self, inputs):

    Z = inputs 

    for layer in self.main_layers:

        Z= layer(Z)

    skip_Z = inputs

    for layer in self.skip_layers:

        skip_Z= LAYER(skip_Z)

    return self.activation(Z + skip_Z)
model_res = keras.models.Sequential()

model_res.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[28, 28, 1], padding='same', use_bias=False))

model_res.add(keras.layers.BatchNormalization())

model_res.add(keras.layers.Activation('relu'))

model_res.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

prev_filters = 64

for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:

    strides = 1 if filters == prev_filters else 2

    model_res.add(ResidualUnit(filters, strides=strides))

    prev_filters = filters

model_res.add(keras.layers.GlobalAvgPool2D())

model_res.add(keras.layers.Flatten())

model_res.add(keras.layers.Dense(10, activation='softmax'))
model_res.summary()
n_epochs = 50

batch_size = 32

model_res.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

model_res.fit(X_train_net, y_train_net, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val))
pd.DataFrame(history.history).plot(figsize=(8,5))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()
model_res.evaluate(X_test_net, y_test_net)
X_submit = test.astype('float32')

X_submit /= 255.0

X_submit = X_submit.values.reshape(-1,28,28,1)
y_pred = model_cnn.predict(X_submit)
y_pred_final = np.argmax(y_pred, axis=1)

y_pred_final
submission = pd.DataFrame()

submission['ImageId'] = pd.Series(range(1,28001))

submission['Label'] = y_pred_final

submission.head()

submission.to_csv('submit.csv', index=False)