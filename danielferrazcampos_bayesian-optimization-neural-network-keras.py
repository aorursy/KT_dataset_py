from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials



from sklearn.datasets import make_classification

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier



from sklearn.metrics import roc_auc_score

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import Adadelta, Adam, rmsprop



from collections import Counter

import pandas as pd

import numpy as np

import pickle

import time

import sys



seed = 42 # Set seed for reproducibility purposes

metric = 'accuracy' # See other options https://scikit-learn.org/stable/modules/model_evaluation.html

kFoldSplits = 5



np.random.seed(seed) # Set numpy seed for reproducibility



# Create a toy-dataset using make_classification function from scikit-learn

X,Y=make_classification(n_samples=10000,

                        n_features=30,

                        n_informative=2,

                        n_redundant=10,

                        n_classes=2,

                        random_state=seed)



# Split in train-test-validation datasets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.25, random_state=seed) # 0.25 x 0.8 = 0.2



# Check on created data

print("Training features size:   %s x %s\nTesting features size:    %s x %s\nValidation features size: %s x %s\n" % (X_train.shape[0],X_train.shape[1], 

                                                                                                                     X_test.shape[0],X_test.shape[1], 

                                                                                                                     X_validation.shape[0],X_validation.shape[1]))



# Create a function to print variable name

def namestr(obj, namespace = globals()):

    return [name for name in namespace if namespace[name] is obj]



# Check on class distribution

for x in [Y_train, Y_test, Y_validation]:

    print(namestr(x)[0])

    counter = Counter(x)

    for k,v in counter.items():

        pct = v / len(x) * 100

        print("Class: %1.0f, Count: %3.0f, Percentage: %.1f%%" % (k,v,pct))

    print("")
X = X_train

y = Y_train

X_val = X_test

y_val = Y_test
units_options = np.arange(32, 1024 + 1, 32, dtype=int)

dropout_options = np.arange(.20,.75 + 0.01, 0.025, dtype=float)

batchsize_options = np.arange(32, 128 + 1, 32, dtype=int)
space = {'choice': hp.choice('num_layers',

                            [ {'layers':'two', },

                              {'layers':'three',

                                    'units3': hp.choice('units3', units_options), 

                                    'dropout3': hp.choice('dropout3', dropout_options)}

                            ]),



            'units1': hp.choice('units1', units_options),

            'units2': hp.choice('units2', units_options),



            'dropout1': hp.choice('dropout1', dropout_options),

            'dropout2': hp.choice('dropout2', dropout_options),



            'batch_size' : hp.choice('batch_size', batchsize_options),



            'nb_epochs' :  10,

            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),

            'activation': 'relu'

        }
def f_nn(params):   



    model = Sequential()

    model.add(Dense(units=params['units1'], input_dim = X.shape[1])) 

    model.add(Activation(params['activation']))

    model.add(Dropout(params['dropout1']))



    model.add(Dense(units=params['units2'], kernel_initializer = "glorot_uniform")) 

    model.add(Activation(params['activation']))

    model.add(Dropout(params['dropout2']))



    if params['choice']['layers']== 'three':

        model.add(Dense(units=params['choice']['units3'], kernel_initializer = "glorot_uniform")) 

        model.add(Activation(params['activation']))

        model.add(Dropout(params['choice']['dropout3']))    



    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])



    model.fit(X, y, epochs=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)



    pred_auc = model.predict_proba(X_val, batch_size = 128, verbose = 0)

    acc = roc_auc_score(y_val, pred_auc)

    print("AUC: %.5f" % (acc))



    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()

best = fmin(f_nn, space, algo=tpe.suggest, max_evals=5, trials=trials)

print('\nBest params found:\n', best)