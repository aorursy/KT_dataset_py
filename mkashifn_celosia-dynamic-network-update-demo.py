# Imports

import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf

from tensorflow.python.client import device_lib

import time

import seaborn as sns

import matplotlib.gridspec as gridspec



from keras.layers import Input, Dense

from keras import regularizers, Model

from keras.models import Sequential, model_from_json

from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import RMSprop



from sklearn.manifold import TSNE

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, precision_recall_curve, classification_report, confusion_matrix, precision_score, average_precision_score, roc_curve, auc, recall_score, f1_score
# Import benign and gafgyt combo dataset for a given device id (1-9)

def import_dataset_benign_gagfyt_combo(device_id):

    normal = pd.read_csv('../input/nbaiot-dataset/{}.benign.csv'.format(device_id))

    n_X = normal.iloc[:,]

    n_X_scaled = MinMaxScaler().fit_transform(n_X.values)

    n_y = np.ones(n_X.shape[0]) # 1 represents normal



    anomalous = pd.read_csv('../input/nbaiot-dataset/{}.gafgyt.combo.csv'.format(device_id))

    a_X = anomalous.iloc[:,]

    a_X_scaled = MinMaxScaler().fit_transform(a_X.values)

    a_y = np.zeros(a_X.shape[0]) # 0 represents anomalous



    #normal.info()

    #normal.describe()

    #normal.head()



    #anomalous.info()

    #anomalous.describe()

    #anomalous.head()



    return (n_X_scaled, n_y, a_X_scaled, a_y)
def build_model(feature_count):

    model = Sequential()

    model.add(Dense(units=8, kernel_initializer="uniform", activation="relu", input_dim=feature_count)) # Hidden Layer 1 with 8 nodes

    model.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))  # Hidden Layer 2 with 6 nodes

    model.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid")) # Output Layer

    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
def change_model_feature_count(model, feature_count):

    # replace input shape of first layer

    model._layers[1].batch_input_shape = (None, feature_count)



    # rebuild model architecture by exporting and importing via json

    new_model = model_from_json(model.to_json())

    #new_model.summary()



    # copy weights from old model to new one

    for layer in new_model.layers:

        try:

            layer.set_weights(model.get_layer(name=layer.name).get_weights())

        except:

            print("Could not transfer weights for layer {}".format(layer.name))



    new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return new_model
def train_model(model, X_train, y_train):

    history = model.fit(X_train, y_train,

                    batch_size=64,

                    epochs=3) #,

                    # We pass some validation for

                    # monitoring validation loss and metrics

                    # at the end of each epoch

                    #validation_data=(X_test, y_test))
def evaluate_model(model, X_test, y_test):

    #model.summary()



    y_pred = model.predict(X_test)

    y_pred = y_pred.round()



    #print ("")

    #print ("Classification Report: ")

    #print (classification_report(y_test, y_pred.round()))



    #print ("")

    #print ("Accuracy Score: ", accuracy_score(y_test, y_pred.round()))

    #loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    return (accuracy, precision, recall, f1)
def limit_dataset_feature_count(feature_count, n_X_scaled, n_y, a_X_scaled, a_y):

    X = np.append(n_X_scaled[:,:feature_count], a_X_scaled[:,:feature_count], axis = 0)

    y = np.append(n_y, a_y)



    return train_test_split(X, y, test_size=0.25)
def print_accuracy_metrices(title, metrices):

    (accuracy, precision, recall, f1) = metrices

    print (f"{title}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
def dynamic_network_expansion(device_id):

    devices_names = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor', 'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']

    (n_X_scaled, n_y, a_X_scaled, a_y) = import_dataset_benign_gagfyt_combo(device_id)

    feature_count = 100

    X_train, X_test, y_train, y_test = limit_dataset_feature_count(feature_count, n_X_scaled, n_y, a_X_scaled, a_y)



    model = build_model(feature_count)



    train_model(model, X_train, y_train)



    evaluate_model(model, X_test, y_test)



    new_feature_count = 115

    X_train, X_test, y_train, y_test = limit_dataset_feature_count(new_feature_count, n_X_scaled, n_y, a_X_scaled, a_y)

    new_model = change_model_feature_count(model, new_feature_count)



    before = evaluate_model(new_model, X_test, y_test)

    train_model(new_model, X_train, y_train)

    after = evaluate_model(new_model, X_test, y_test)



    return (devices_names[device_id - 1], before, after)
device_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

metrices = []

for device_id in device_ids:

    metrices.append(dynamic_network_expansion(device_id))
for metric in metrices:

    (name, before, after) = metric

    print(name)

    print("")

    print_accuracy_metrices(f"{name}-Before", before)

    print_accuracy_metrices(f"{name}-After", after)

    print("-----------------")