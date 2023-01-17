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
import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPClassifier
def load_dataset():

    ds = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

    y = np.array(ds.label).ravel()

    X = np.array(ds.iloc[:, 1:])

    return X, y
def visualize_digits(X):

    plt.figure(figsize=(12,10))

    for num in range(0, 20):

        plt.subplot(5, 4, num+1)

        data = X[num].reshape(28, 28)

        plt.imshow(data, cmap='afmhot', interpolation='None')
def test_model(model, X_test):

    return model.predict(X_test)
def get_train_test_split(X, y):

    return train_test_split(X, y, test_size=.1)
def train_model(X_train, y_train, model_name):

    if model_name == 'decision_tree':

        model = DecisionTreeClassifier()

        model.fit(X_train, y_train)

    elif model_name == 'random_forest':

        model = RandomForestClassifier()

        model.fit(X_train, y_train)

    return model
def get_kaggle_test_results(model, changeshape):

    test_ds = pd.read_csv("data/digit-recognizer/test.csv")

    test = np.array(test_ds)

    # Apply scaling

    test = test/255

    

    # For CNN

    if changeshape == True:

        test = test.reshape(test.shape[0], 28, 28)

    

    predictions = test_model(model, test)

    # For CNN

    if changeshape == True:

        pred = [np.argmax(x) for x in predictions]    



    result = pd.DataFrame(data={'ImageId':np.arange(1, len(predictions)+1), 'Label':pred}).set_index('ImageId')

    result.to_csv("data/digit-recognizer/kaggle_dt.csv")
# Load dataset into X, y

X, y = load_dataset()
# Visualize data

visualize_digits(X)
# Split dataset

X_train, X_test, y_train, y_test = get_train_test_split(X, y)
DT = train_model(X_train, y_train, 'decision_tree')

DT_predict = test_model(DT, X_test)

accuracy_score(y_test, DT_predict)
RF = train_model(X_train, y_train, 'random_forest')

RF_predict = RF.predict(X_test)

accuracy_score(y_test, RF_predict)
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(solver='lbfgs', alpha=0.00005, hidden_layer_sizes=(2500,), random_state=1)

NN.fit(X_train/255, y_train)
NN_predict = NN.predict(X_test)

accuracy_score(y_test, NN_predict)
import tensorflow as tf

from tensorflow import keras
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_train = X_train/255.0

X_test = X_test.reshape(X_test.shape[0], 28, 28)

X_test = X_test/255.0
model.fit(X_train, y_train, epochs=20)
cnn_predictions = model.predict(X_test)

pred = [np.argmax(x) for x in cnn_predictions]

accuracy_score(y_test, pred)