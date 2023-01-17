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

from keras.models import Sequential



from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, precision_recall_curve, classification_report, confusion_matrix, average_precision_score, roc_curve, auc
# Make sure that the dataset exists

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))
# Importing benign gafgyt combo dataset for a given device id (1-9)

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
# AutoEncoder



def generate_and_train_autoencoder(X):

    ## input layer 

    input_layer = Input(shape=(X.shape[1],))



    ## encoding part

    encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)

    encoded = Dense(50, activation='relu')(encoded)



    ## decoding part

    decoded = Dense(50, activation='tanh')(encoded)

    decoded = Dense(100, activation='tanh')(decoded)



    ## output layer

    output_layer = Dense(X.shape[1], activation='relu')(decoded)



    autoencoder = Model(input_layer, output_layer)

    autoencoder.compile(optimizer="adadelta", loss="mse")



    autoencoder.fit(X[0:1000], X[0:1000], 

                    batch_size = 256, epochs = 10, 

                    shuffle = True, validation_split = 0.20);



    return autoencoder
# hidden representation

def get_hidden_representation_normal_anomalous(autoencoder, n_X, a_X):

    hidden_representation = Sequential()

    hidden_representation.add(autoencoder.layers[0])

    hidden_representation.add(autoencoder.layers[1])

    hidden_representation.add(autoencoder.layers[2])



    normal_hid_rep = hidden_representation.predict(n_X[:3000])

    anomalous_hid_rep = hidden_representation.predict(a_X[:3000])



    return (normal_hid_rep, anomalous_hid_rep)
def tsne_plot(x, y, title="Scatter Plot", name="graph.png"):

    tsne = TSNE(n_components=2, random_state=0)

    X_t = tsne.fit_transform(x)



    plt.figure(figsize=(12, 8))

    plt.scatter(X_t[np.where(y == 1), 0], X_t[np.where(y == 1), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Normal')

    plt.scatter(X_t[np.where(y == 0), 0], X_t[np.where(y == 0), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Anomalous')



    plt.title(title)

    plt.legend(loc='best');

    plt.savefig('{}-{}'.format(title,name));

    plt.show();
# Combine normal and anomalous data

def combine_normal_anomalous(normal, anomalous):

    X = np.append(normal, anomalous, axis = 0)

    y_n = np.ones(normal.shape[0])

    y_a = np.zeros(anomalous.shape[0])

    y = np.append(y_n, y_a)



    return (X, y)
# Evaluate model's performace

def evaluate_model(X, y):

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

    clf = LogisticRegression(solver="lbfgs").fit(train_X, train_y)

    pred_y = clf.predict(test_X)

    return (test_y, pred_y)
# Show model's report

def show_model_report(title, X, y, hid_X, hid_y, test_y, pred_y):

    #tsne_plot(X, y, '{}-original'.format(title), "original.png")

    tsne_plot(hid_X, hid_y, title, "hidden_representation.png")



    print(title)

    print ("")

    print ("Classification Report: ")

    print (classification_report(test_y, pred_y))



    print ("")

    print ("Accuracy Score: ", accuracy_score(test_y, pred_y))
# Detect zero-day attack

def detect_zero_day_attack(device_name, device_id):

    (n_X_scaled, n_y, a_X_scaled, a_y) = import_dataset_benign_gagfyt_combo(device_id)

    autoencoder = generate_and_train_autoencoder(n_X_scaled)

    (normal_hid_rep, anomalous_hid_rep) = get_hidden_representation_normal_anomalous(autoencoder, n_X_scaled, a_X_scaled)

    (X, y) = combine_normal_anomalous(n_X_scaled, a_X_scaled)

    (hid_X, hid_y) = combine_normal_anomalous(normal_hid_rep, anomalous_hid_rep)

    (test_y, pred_y) = evaluate_model(hid_X, hid_y)

    show_model_report(device_name, X, y, hid_X, hid_y, test_y, pred_y)
devices = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor', 'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']

#devices = ['Danmini_Doorbell', 'Ecobee_Thermostat']

for device_id, device_name in enumerate(devices, 1):

    detect_zero_day_attack(device_name, device_id)
