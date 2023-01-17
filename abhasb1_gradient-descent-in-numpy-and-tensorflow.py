import numpy as np

import pandas as pd

import io



def prepare_data(df):

    features = df.drop(['Serial No.'], axis=1)

    labels = df['Chance of Admit ']



    X = df.drop(['Research', 'Serial No.'], axis=1).values/100

    Y = df[['Chance of Admit ']].values

  

    return X, Y



df = pd.read_csv('../input/admission/Admission_Predict_Ver1.1.csv', encoding='utf-8')

df.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt



df = pd.read_csv('../input/admission/Admission_Predict_Ver1.1.csv', encoding='utf-8')

X, Y = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



minmax = MinMaxScaler()

X_train = minmax.fit_transform(X_train)

X_test = minmax.transform(X_test)



def _np_predictions(X, weights, bias):

    return np.dot(X, weights)



def _np_cost_function(predictions, y):

    error = predictions - y

    cost = (1/2)*(error**2).mean()

    return cost



def _np_gradients(X, predictions, y):

    error = predictions - y

    grad = np.dot(X.T, error)/X.shape[0]

    bias_grad = np.sum(error)/X.shape[0]

    return grad, bias_grad



num_features = X_train.shape[1]

weights = np.random.normal(size=(num_features, 1))

bias = 0



losses = []

epoch = []

eta = 0.1

n_iter = 20



for i in range(n_iter):

    predictions = _np_predictions(X_train, weights, bias)

    losses.append(_np_cost_function(predictions, y_train))

    epoch.append(i+1)



    grad, bias_grad = _np_gradients(X_train, predictions, y_train)

    weights -= eta*grad

    bias -= eta*bias_grad



plt.plot(epoch, losses);
#Evaluate

from sklearn.metrics import mean_squared_error

print(f"Mean Squared Error after {n_iter} iterations: {np.round(mean_squared_error(y_test, _np_predictions(X_test, weights, bias)),2)}")
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



df = pd.read_csv('../input/admission/Admission_Predict_Ver1.1.csv', encoding='utf-8')

X, Y = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



X_train = tf.constant(X_train, dtype=tf.double)

y_train = tf.constant(y_train, dtype=tf.double)

X_test = tf.constant(X_test, dtype=tf.double)

y_test = tf.constant(y_test, dtype=tf.double)





def _tf_predictions(X, weights,bias):

      return tf.tensordot(X, weights, axes=1) + bias



def _tf_cost_function(predictions, y):

    error = predictions - y

    squared_error = tf.square(predictions - y)

    cost = tf.reduce_mean(squared_error)

    return cost



def _tf_gradients(X, predictions, y):

    error = predictions - y

    grad = tf.tensordot(tf.transpose(X), error, axes=1)/X.shape[0]

    bias_grad = tf.reduce_mean(error)

    return grad, bias_grad





num_features = X_train.shape[1]

bias = 0

weights = tf.random.normal(shape=(num_features, 1), dtype=tf.double)



losses = []

epoch = []

eta = 0.01

n_iter = 20



for i in range(n_iter):

    predictions = _tf_predictions(X_train, weights, bias)

    losses.append(_tf_cost_function(predictions, y_train))

    epoch.append(i+1)



    grad, bias_grad = _tf_gradients(X_train, predictions, y_train)

    weights -= eta*grad

    bias -= eta*bias_grad



plt.plot(epoch, losses);
#Evaluate

from sklearn.metrics import mean_squared_error

print(f"Mean Squared Error after {n_iter} iterations: {np.round(mean_squared_error(y_test, _np_predictions(X_test, weights, bias)),2)}")
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



df = pd.read_csv('../input/admission/Admission_Predict_Ver1.1.csv', encoding='utf-8')

X, Y = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



X_train = tf.constant(X_train, dtype=tf.double)

y_train = tf.constant(y_train, dtype=tf.double)

X_test = tf.constant(X_test, dtype=tf.double)

y_test = tf.constant(y_test, dtype=tf.double)





# --------------------------------------

def _tf_predictions(X, weights,bias):

    return tf.tensordot(X, weights, axes=1) + bias



def _tf_cost_function(predictions, y):

    error = predictions - y

    squared_error = tf.square(predictions - y)

    cost = tf.reduce_mean(squared_error)

    return cost



def _tf_gradients(X, predictions, y):

    error = predictions - y

    grad = tf.tensordot(tf.transpose(X), error, axes=1)/X.shape[0]

    bias_grad = tf.reduce_mean(error)

    return grad, bias_grad



#---------------------------------------

## Initialize some hyperparams



num_samples = X.shape[0]

num_features = X.shape[1]

weights = tf.random.normal((num_features, 1), dtype=tf.double)

bias = 0



num_epochs = 20

batch_size = 10

learning_rate = 0.01



epochs = []

losses = []



## create a tf.data.Dataset object to store and batch the data efficiently



dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

dataset = dataset.shuffle(num_samples)

dataset = dataset.repeat(num_epochs)

dataset = dataset.batch(batch_size )

iterator = dataset.__iter__()



# ---------------------------------------



for i in range(num_epochs):

    epoch_loss = []

    for b in range(int(num_samples / batch_size)):

        try:

            X_batch, y_batch = iterator.get_next()

            predictions = _tf_predictions(X_batch, weights, bias)

            epoch_loss.append(_tf_cost_function(predictions, y_batch))



            grad, bias_grad = _tf_gradients(X_batch, predictions, y_batch)

            weights -= learning_rate*grad

            bias -= learning_rate*bias_grad

        except tf.errors.OutOfRangeError:

            pass

  

    losses.append(np.array(epoch_loss).mean())

    epochs.append(i+1)





plt.plot(epochs, losses);
#Evaluate

from sklearn.metrics import mean_squared_error

print(f"Mean Squared Error after {num_epochs} iterations: {np.round(mean_squared_error(y_test, _np_predictions(X_test, weights, bias)),2)}")