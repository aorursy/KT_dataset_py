# import base modules

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline
# load training data

print('Load data...')

df_train = pd.read_csv('../input/train.csv')
# let's keep just the boolean features

train = df_train.iloc[:,10:]

# ... and the y

y = df_train.y
# Let's quickly build an Autoencoder with 1 hidden layer and only 2 hidden neurons. 

from keras.layers import Input, Dense

from keras.models import Model



encoding_dim = 2

input_layer = Input(shape=(train.shape[1],))

encoded = Dense(encoding_dim, activation='relu')(input_layer)

decoded = Dense(train.shape[1], activation='sigmoid')(encoded)



# let's create and compile the autoencoder

autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# let's train the autoencoder, checking the progress on a validation dataset 

from sklearn.model_selection import train_test_split

X1, X2, Y1, Y2 = train_test_split(train, train, test_size=0.2, random_state=42)



# these parameters seems to work for the Mercedes dataset

autoencoder.fit(X1.values, Y1.values,

                epochs=300,

                batch_size=200,

                shuffle=False,

                verbose = 2,

                validation_data=(X2.values, Y2.values))
# now let's evaluate the coding of the initial features

encoder = Model(input_layer, encoded)

preds = encoder.predict(train.values)
#... and let's plot the two components of the compression on a scatter plot that also shows 

# the y value associated to each point. PCA decomposition is provided as well for comparison

plt.figure(figsize = (17,5))

plt.subplot(131)

plt.scatter(preds[:,0],preds[:,1],  c = y, cmap = "RdGy", 

            edgecolor = "None", alpha=1, vmin = 75, vmax = 150)

plt.colorbar()

plt.title('AE Scatter Plot')





# ICA and PCA (first 2 components)

from sklearn.decomposition import PCA, FastICA # Principal Component Analysis module

ica = FastICA(n_components=2)

ica_2d = ica.fit_transform(train.values)



plt.subplot(132)

plt.scatter(ica_2d[:,0],ica_2d[:,1],  c = y, cmap = "RdGy",

            edgecolor = "None", alpha=1, vmin = 75, vmax = 150)

plt.colorbar()

plt.title('ICA Scatter Plot')



pca = PCA(n_components=2)

pca_2d = pca.fit_transform(train.values)



plt.subplot(133)

plt.scatter(pca_2d[:,0],pca_2d[:,1],  c = y, cmap = "RdGy",

            edgecolor = "None", alpha=1, vmin = 75, vmax = 150)

plt.colorbar()

plt.title('PCA Scatter Plot')



plt.show()
var = df_train.X232

plt.figure(figsize = (15,5))

plt.subplot(131)

plt.scatter(preds[:,0],preds[:,1],  c = y, cmap = "RdGy", 

            edgecolor = "None", alpha=1, vmin = 75, vmax = 150)

plt.title('AE Scatter Plot')



plt.subplot(132)

plt.scatter(preds[:,0],preds[:,1],  c = var, cmap = "jet", 

            edgecolor = "None", alpha=1, vmin = 0, vmax = 1)

plt.title('X232')



plt.subplot(133)

bins = np.linspace(75, 275, 51)

plt.hist(y[var==0], bins, alpha=0.5, label='0', color = plt.cm.jet(0))

plt.hist(y[var==1], bins, alpha=0.75, label='1', color = plt.cm.jet(255))

plt.title('X232')

plt.legend(loc='upper right')



plt.show()