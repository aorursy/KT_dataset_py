# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import keras as keras

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



#from sklearn.model_selection import KFold

#from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

import eli5

from eli5.sklearn import PermutationImportance

import random



from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from sklearn.model_selection import StratifiedKFold
train = pd.read_csv('../input/train.csv', index_col='id')

test = pd.read_csv('../input/test.csv', index_col='id')

sample_sub = pd.read_csv('../input/sample_submission.csv', index_col='id')
# split train in features and labels

X_train = train.iloc[:, 1:]

y_train = train.iloc[:, 0]



X_test = test.iloc[:, :]

y_test = sample_sub.iloc[:, 0]
# Number of labels per classes

y_train.value_counts()
# Number of labels per classes

max_samples = np.max(y_train.value_counts())

class_weight = y_train.value_counts()/max_samples

class_weight = class_weight.to_dict()

class_weight
from sklearn.metrics import roc_auc_score

import time



def train_model(model, X, y, X_test, \

                folds=None, \

                averaging='usual', \

                model_type='sklearn', \

                tf_session=None, \

                epochs=10, \

                class_weight=None):

    n_fold = 10

    if(folds is None):

        folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

    

    oof = np.zeros(len(X))

    prediction = np.zeros(len(X_test))

    scores = []

    feature_importance = pd.DataFrame()

    

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        #print('Fold', fold_n, 'started at', time.ctime())

        #print(train_index, valid_index)

        X_train, X_valid = X[train_index], X[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if(model_type == 'sklearn'):

            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid)

            y_pred_valid = y_pred_valid.reshape(-1,)

        elif(model_type == 'tensorflow'):

            model.fit(X_train, y_train, epochs=epochs, class_weight=class_weight)

            y_pred_valid = model.predict(X_valid)

            y_pred_valid = y_pred_valid.eval(session=tf_session)

            y_pred_valid = np.reshape(y_pred_valid, (-1,))

        

        score = roc_auc_score(y_valid, y_pred_valid)

        #print(type(y_valid), type(y_pred_valid) )

        print(f'Fold {fold_n}. AUC: {score:.4f}.')

        #print('')



        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(score)

        

        if(model_type == 'sklearn'):

            y_pred_test = model.predict_proba(X_test)[:, 1]

            if averaging == 'usual':

                prediction += y_pred_test/n_fold

            elif averaging == 'rank':

                prediction += pd.Series(y_pred_test).rank().values  

        elif(model_type == 'tensorflow'):

            y_pred_test = model.predict(X_test)

            y_pred_test = y_pred_test.eval(session=tf_session)

            y_pred_test = np.reshape(y_pred_test, (-1,))

            

            prediction += (y_pred_test/n_fold)

        else:

            prediction = 0

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    return oof, prediction, scores
# Benchmark score

# CV mean score: 0.7635, std: 0.0469.
from sklearn.preprocessing import MinMaxScaler, StandardScaler

minMaxScaler= MinMaxScaler()

scaler = StandardScaler()



X_train = train.iloc[:, 1:]

y_train = train.iloc[:, 0]



X_test = test.iloc[:, :]

y_test = sample_sub.iloc[:, 0]



X_train_scl = scaler.fit_transform(X_train)

X_test_scl = scaler.transform(X_test)



#X_train_scl = minMaxScaler.fit_transform(X_train_scl)

#X_test_scl = minMaxScaler.transform(X_test_scl)
class C3POModel:

    def __init__(self, input_dim=300, debug=False, autoencoder=True):

        #initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

        #initializer = keras.initializers.RandomUniform()

        

        ##### Autoencoder block to get peculiar features in unsupervised manner ####

        X = keras.layers.Input(shape=(input_dim, ), name='Input')

        if(autoencoder):

            Y = keras.layers.Dense(input_dim, input_dim=input_dim, name='encoder_layer_1', 

                                   #kernel_regularizer=keras.regularizers.l1(0.01),

                                   activation='tanh')(X)

            Y = keras.layers.Dense(int(input_dim/3), name='encoder_layer_2', 

                                   #kernel_regularizer=keras.regularizers.l1(0.01),

                                   activation='tanh')(Y)

            #Y = keras.layers.Dense(input_dim, name='encoder_layer_3', 

            #                       activation='tanh')(Y)

            #Y = keras.layers.Dense(150, name='encoder_layer_4', 

            #                       activation='tanh')(Y)

            Y = keras.layers.Dense(input_dim, name='encoder_layer_5', 

                                   #kernel_regularizer=keras.regularizers.l1(0.01),

                                   activation='tanh')(Y)



            #### Average encoded and raw data

            Y = keras.layers.Average()([X, Y])

        else:

            Y = X

        Y = keras.layers.Dense(input_dim, name='layer_1',

                               kernel_regularizer=keras.regularizers.l1_l2(0.01),

                               activation='tanh')(Y)

        #Y = keras.layers.Dropout(0.5)(Y)

        Y = keras.layers.Dense(input_dim*2, name='layer_2', 

                               kernel_regularizer=keras.regularizers.l1_l2(0.01),

                               activation='sigmoid')(Y)

        #Y = keras.layers.Dropout(0.5)(Y)

        Y = keras.layers.Dense(1, name='layer_4', 

                               activation='sigmoid')(Y)

        self.net = keras.Model(X, Y)

        

        self.net.compile(optimizer='adadelta',

              loss=keras.losses.MSE,

              metrics=['accuracy'])

        if(debug):

            print(self.net.summary())

        pass

    

    def fit(self, X, y, epochs=5, verbose=0, validation_split=0.1, class_weight=None):

        callback = None # [

            #keras.callbacks.EarlyStopping(patience=5),

            #keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1)

            #keras.callbacks.ModelCheckpoint(filepath="C3POModel_weights.hdf5", verbose=0, save_best_only=True)

        #]

        history = self.net.fit(X, 

                               y, 

                               epochs=epochs, 

                               verbose=verbose, 

                               validation_split=validation_split,

                               callbacks=callback,

                               class_weight=class_weight

                              )

        return history

    

    def evaluate(self, X, y):

        return self.net.evaluate(X, y)

    

    def predict(self, X):

        #print('called predict()')

        return self.net.call(tf.convert_to_tensor(X, dtype=tf.float32))

    

    def load_weights(self, filepath):

        self.net.load_weights(filepath)

        pass

from keras import backend

backend.clear_session()



with tf.Session() as sess:

    init_op = tf.global_variables_initializer()

    sess.run(init_op)

    model = C3POModel(input_dim=X_train.shape[1], autoencoder=True)

    oof, prediction, scores = train_model(model, \

                                          X_train_scl, \

                                          y_train, \

                                          X_test_scl, \

                                          model_type='tensorflow', \

                                          tf_session=sess, \

                                          class_weight=class_weight)

    

    y_pred = model.predict(X_test_scl)

    y_pred = y_pred.eval(session=sess)

    y_pred = np.reshape(y_pred, (-1,))

    

    prediction = y_pred #(prediction+y_pred)/2

    auc_score = roc_auc_score(y_true=np.append(y_test, (1, 0)), y_score=np.append(np.where(prediction < 0.5, 0, 1), (1, 0)))

    acc_score = accuracy_score(y_true=np.append(y_test, (1, 0)), y_pred=np.append(np.where(prediction < 0.5, 0, 1), (1, 0)))

    

    print(auc_score, acc_score)

    submit = pd.read_csv('../input/sample_submission.csv')

    submit["target"] = prediction

    submit.to_csv("submission.csv", index=False)

    #print(submit.head(30))