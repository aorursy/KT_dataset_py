from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
%matplotlib inline
X_iris = load_iris().data
y_iris = load_iris().target.tolist()

scaler = MinMaxScaler()
scaler.fit(X_iris)
X_iris=scaler.transform(X_iris)
input_dim = 4

#input layer
input_lay = Input(shape=(input_dim,))

#coder
i_1 = Dense(50, activation='relu')(input_lay)
i_2 = Dense(20, activation='relu')(i_1)
i_3 = Dense(10, activation='relu')(i_2)
encoded = Dense(2, activation='linear')(i_3)

#decoder
d_1 = Dense(10, activation='relu')(encoded)
d_2 = Dense(20, activation='relu')(d_1)
d_3 = Dense(50, activation='relu')(d_2)
decoded = Dense(input_dim, activation='linear')(d_3)

autoencoder = Model(input_lay, decoded)
encoder = Model(input_lay, encoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_iris, X_iris, epochs = 500, batch_size = 10)
X_enc = encoder.predict(X_iris)
plt.scatter(X_enc[np.isin(y_iris,0),0],X_enc[np.isin(y_iris,0),1],c='r')
plt.scatter(X_enc[np.isin(y_iris,1),0],X_enc[np.isin(y_iris,1),1],c='b')
plt.scatter(X_enc[np.isin(y_iris,2),0],X_enc[np.isin(y_iris,2),1],c='k')
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X_iris)

plt.scatter(X_pca[np.isin(y_iris,0),0],X_pca[np.isin(y_iris,0),1],c='r')
plt.scatter(X_pca[np.isin(y_iris,1),0],X_pca[np.isin(y_iris,1),1],c='b')
plt.scatter(X_pca[np.isin(y_iris,2),0],X_pca[np.isin(y_iris,2),1],c='k')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1]*X_train.shape[2]))/255.
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1]*X_test.shape[2]))/255.

X = np.concatenate((X_train,X_test))
y = np.concatenate((y_train,y_test))

X_mnist = X[np.isin(y,(0,2,4,8))]
y_mnist = y[np.isin(y,(0,2,4,8))]
input_dim = 784

#input layer
input_lay = Input(shape=(input_dim,))

#coder
i_1 = Dense(1000, activation='relu')(input_lay)
i_2 = Dense(500, activation='relu')(i_1)
i_3 = Dense(100, activation='relu')(i_2)
encoded = Dense(2, activation='linear')(i_3)

#decoder
d_1 = Dense(100, activation='relu')(encoded)
d_2 = Dense(500, activation='relu')(d_1)
d_3 = Dense(1000, activation='relu')(d_2)
decoded = Dense(input_dim, activation='linear')(d_3)

autoencoder = Model(input_lay, decoded)
encoder = Model(input_lay, encoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_mnist, X_mnist, epochs = 500, batch_size = 300)
X_enc = encoder.predict(X_mnist)

plt.scatter(X_enc[np.isin(y_mnist,0),0],X_enc[np.isin(y_mnist,0),1],c='r')
plt.scatter(X_enc[np.isin(y_mnist,2),0],X_enc[np.isin(y_mnist,2),1],c='b')
plt.scatter(X_enc[np.isin(y_mnist,4),0],X_enc[np.isin(y_mnist,4),1],c='k')
plt.scatter(X_enc[np.isin(y_mnist,8),0],X_enc[np.isin(y_mnist,8),1],c='c')
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X_mnist)

plt.scatter(X_pca[np.isin(y_mnist,0),0],X_pca[np.isin(y_mnist,0),1],c='r')
plt.scatter(X_pca[np.isin(y_mnist,2),0],X_pca[np.isin(y_mnist,2),1],c='b')
plt.scatter(X_pca[np.isin(y_mnist,4),0],X_pca[np.isin(y_mnist,4),1],c='k')
plt.scatter(X_pca[np.isin(y_mnist,8),0],X_pca[np.isin(y_mnist,8),1],c='c')