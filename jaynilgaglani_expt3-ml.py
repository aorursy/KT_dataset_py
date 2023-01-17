import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline
from keras.layers import Input, Dense

from keras.models import Model

import warnings

warnings.filterwarnings("ignore")
# from google.colab import drive

# drive.mount('/content/drive')

# from google.colab import files

# uploaded = files.upload()
df = pd.read_csv('../input/wholesale-customers-data-set/Wholesale customers data.csv')



X = df.iloc[:,1:] # Features

y = df.iloc[:,:-7] # Target variable

print(X)

print(y)
normalized_X = preprocessing.normalize(X)
X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.2)
encoding_dim = 4

input_dim = 7



# this is our input placeholder

input_img = Input(shape=(input_dim,))

# "encoded" is the encoded representation of the input

encoded = Dense(encoding_dim)(input_img)

# "decoded" is the lossy reconstruction of the input

decoded = Dense(input_dim)(encoded)



# this model maps an input to its reconstruction

autoencoder = Model(input_img, decoded)



# this model maps an input to its encoded representation

encoder = Model(input_img, encoded)



# create a placeholder for an encoded (2-dimensional) input

encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model

decoder_layer = autoencoder.layers[-1]

# create the decoder model

decoder = Model(encoded_input, decoder_layer(encoded_input))



autoencoder.compile(loss='mean_squared_logarithmic_error', optimizer='adam',metrics=['accuracy'])



autoencoder.fit(X_train, X_train,

                epochs=150,

                batch_size=40,

                shuffle=True,

                validation_data=(X_test, X_test))



# encode and decode some data points

# note that we take them from the *test* set

encoded_datapoints = encoder.predict(X_test)

decoded_datapoints = decoder.predict(encoded_datapoints)
# print('Original Datapoints :')

# print(X_test)

# print('Reconstructed Datapoints :')

# print(decoded_datapoints)
from sklearn.cluster import KMeans
# Display the results of the clustering from implementation for 2 clusters

clusterer = KMeans(n_clusters = 2 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state=111, algorithm='elkan')

clusterer.fit(decoded_datapoints)

preds = clusterer.predict(decoded_datapoints)

centers = clusterer.cluster_centers_

print(preds)
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

le = LabelEncoder()

new_y = le.fit_transform(y_test)

print(accuracy_score(new_y,preds))

print(confusion_matrix(new_y,preds))

print(classification_report(new_y,preds))
# for i in range(len(preds)):

#     print(str(new_y[i])+" : "+str(preds[i]))
plt.scatter(decoded_datapoints[:,0],decoded_datapoints[:,1],c=clusterer.labels_,cmap='viridis')

plt.scatter(clusterer.cluster_centers_[:,0],clusterer.cluster_centers_[:,1],marker='p',c='r',linewidths=7)

plt.xlabel("Cluster Coefficients")

plt.title("KMeans Clustering")

plt.ylabel("Clustering Values")

plt.show()