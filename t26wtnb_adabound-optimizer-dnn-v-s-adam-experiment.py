!pip install keras-adabound
import numpy as np

import tensorflow as tf

import pandas as pd

import os

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras import Sequential

from keras import layers

from keras import backend as K

from keras.layers.core import Dense

from keras import regularizers

from keras.layers import Dropout

from keras.constraints import max_norm

from keras.callbacks import EarlyStopping



from keras_adabound import AdaBound



pd.set_option('display.max_columns', 200)
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
train_df.target.value_counts()
X_train = train_df.drop(["target", "ID_code"], axis=1)

X_test = test_df.drop(["ID_code"], axis=1)

y_train = train_df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=71)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape, X_test.shape
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_valid = sc.transform(X_valid)

X_test = sc.transform(X_test)
# Add RUC metric to monitor NN

def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
# callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
input_dim = X_train.shape[1]
def create_model():

    model = Sequential()

    

    # Input layer

    model.add(Dense(units = 200, activation = "relu", input_dim = input_dim, kernel_initializer = "normal", kernel_regularizer=regularizers.l2(0.005), 

                    kernel_constraint = max_norm(5.)))

    # Add dropout regularization

    model.add(Dropout(rate=0.2))



    # Hidden layer

    model.add(Dense(units = 200, activation='relu', kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))

    # Add dropout regularization

    model.add(Dropout(rate=0.2))



    # Stacking hidden layers

    for _ in range(7):

        # Hidden layer

        model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))

        # Add dropout regularization

        model.add(Dropout(rate=0.3))



    # Hidden layer

    model.add(Dense(200, activation='tanh', kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))

    # Add dropout regularization

    model.add(Dropout(rate=0.4))



    # Output layer

    model.add(layers.Dense(units = 1, activation='sigmoid'))



    return model
model_adabound = create_model()

model_adabound.summary()
model_adabound.compile(loss='binary_crossentropy', optimizer=AdaBound(lr=1e-3, final_lr=0.1), metrics=['accuracy', auc])
history_adabound = model_adabound.fit(X_train, y_train, batch_size=128, epochs=100, validation_data=(X_valid, y_valid)) #, callbacks=[callback])
y_pred = model_adabound.predict_proba(X_valid)

roc_auc_score(y_valid, y_pred)
pred = model_adabound.predict(X_test)

pred_ = pred[:,0]
pred_
sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub_df["target"] = pred_

sub_df.head()
sub_df.to_csv('Santander_submit_simple_DNN_AdaBound.csv', index=False)
model_adam = create_model()

model_adam.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])

history_adam = model_adam.fit(X_train, y_train, batch_size=128, epochs=100, validation_data=(X_valid, y_valid)) #, callbacks=[callback])
y_pred_adam = model_adam.predict_proba(X_valid)

roc_auc_score(y_valid, y_pred_adam)
def plot_history(history1, history2, history1_label, history2_label, metric):

    fig = plt.figure(figsize=(10, 10))

    fig.patch.set_facecolor("white")

    

    plt.plot(history1.history[metric], label='{} {}'.format(history1_label, metric))

    plt.plot(history1.history['val_{}'.format(metric)], label='{} val_{}'.format(history1_label, metric))

    plt.plot(history2.history[metric], label='{} {}'.format(history2_label, metric))

    plt.plot(history2.history['val_{}'.format(metric)], label='{} val_{}'.format(history2_label, metric))

    plt.title(metric)

    plt.xlabel('epoch')

    plt.ylabel(metric)

    plt.legend()

    plt.show()
plot_history(history_adam, history_adabound, 'Adam', 'AdaBound', 'loss')
plot_history(history_adam, history_adabound, 'Adam', 'AdaBound', 'auc')