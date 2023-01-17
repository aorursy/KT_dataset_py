
from sklearn.model_selection import train_test_split
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import numpy as np
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


def data_power_consumption(path_to_dataset,
                           sequence_length=50,
                           ratio=1.0):

    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=";")
        power = []
        count = 0
        for line in data:
            try:
                power.append(float(line[2]))
                count += 1
            except ValueError:
                pass
            # 2049280.0 is the total number of valid values, i.e. ratio = 1.0
            if count > 100000:
                break

    print ("Data loaded from csv. Formatting...")

    result = []
    for index in range(len(power) - sequence_length):
        result.append(power[index: index + sequence_length])
    result = np.array(result)  # shape (2049230, 50)

    mean = result.mean()
    result -= mean

    row = int(round(0.9 * result.shape[0]))
    X = result[:,:-1]
    Y = result[:,-1]

    X_train,X_test,y_train,y_test = train_test_split(X,Y,train_size=.2)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test]


def build_model():
    model = Sequential()
    layers = [1, 50, 100, 1]

    model.add(LSTM(
        layers[1],
        input_shape=(49,1),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        layers[3]))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam",metrics=['accuracy'])
    return model


def run_network():
    epochs = 1
    ratio = 0.5
    sequence_length = 50
    path_to_dataset = '../input/household_power_consumption.txt'

    X_train, y_train, X_test, y_test = data_power_consumption(
            path_to_dataset, sequence_length, ratio)

    print ('\nData Loaded. Compiling...\n')

    model = build_model()

    model.fit(
           X_train, y_train,
           batch_size=512, nb_epoch=epochs, validation_split=0.05)
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted,y_test

predicted,y_test = run_network()
try:
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        ax.plot(y_test[:200],'r')
        plt.plot(predicted[:200],'b')
        plt.show()
except Exception as e:
    print (str(e))
