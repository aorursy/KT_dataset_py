import pandas as pd

import numpy as np

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from sklearn.pipeline import make_pipeline
SEED = 42

from tensorflow.random import set_seed

from numpy.random import seed

seed(SEED)

set_seed(SEED)
df = pd.read_csv('../input/mines-vs-rocks/sonar.all-data.csv', header = None)

df = df.values

X = df[:,0:60].astype(float)

Y = df[:,60]

print ('X Shape :', X.shape)

print ('Y Shape :', Y.shape)

print ('Number of Unique Values in Y:', set(Y))
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

Y_encoded = encoder.fit_transform(Y).astype(int)

print ('Shape of Y_encoded :', len(Y_encoded))

print ('Unique values in Y_encoded :', list(set(Y_encoded)))

print ('Inverse transforming : ', encoder.inverse_transform(list(set(Y_encoded))))
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_transformed = ss.fit_transform(X)

X_transformed.shape
def baseline_model():

    

    model = Sequential()

    model.add(Dense(60, input_dim=(60), activation = 'relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    

    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

    

    return model
single_model = baseline_model()

%time history = single_model.fit(X_transformed, Y_encoded, epochs = 200, batch_size = 8, verbose = 0, validation_split = 0.1)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.show()



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.show()
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
EPOCHS     = 50

BATCH_SIZE = 8

VERBOSE    = 0

FOLDS      = 10
kfold = StratifiedKFold(n_splits = FOLDS, shuffle = True, random_state = SEED)

estimators = make_pipeline(StandardScaler(), KerasClassifier(build_fn = baseline_model, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = VERBOSE))

results = cross_val_score(estimators, X, Y_encoded, cv = kfold)

print (f'Mean Accuracy : {round(results.mean()*100,2)} %, Std. dev : {round(results.std()*100,2)}%')
%%time 

def small_model():

    model = Sequential()

    model.add(Dense(30, input_dim=(60), activation = 'relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

    return model



kfold = StratifiedKFold(n_splits = FOLDS, shuffle = True, random_state = SEED)

estimators = make_pipeline(StandardScaler(), KerasClassifier(build_fn = small_model, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = VERBOSE))

results = cross_val_score(estimators, X, Y_encoded, cv = kfold)

print (f'Mean Accuracy : {round(results.mean()*100,2)} %, Std. dev : {round(results.std()*100,2)}%')
%%time 

def large_model():

    model = Sequential()

    model.add(Dense(60, input_dim=(60), activation = 'relu'))

    model.add(Dense(60, activation = 'relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

    return model



kfold = StratifiedKFold(n_splits = FOLDS, shuffle = True, random_state = SEED)

estimators = make_pipeline(StandardScaler(), KerasClassifier(build_fn = large_model, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = VERBOSE))

results = cross_val_score(estimators, X, Y_encoded, cv = kfold)

print (f'Mean Accuracy : {round(results.mean()*100,2)} %, Std. dev : {round(results.std()*100,2)}%')