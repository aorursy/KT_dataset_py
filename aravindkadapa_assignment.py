from IPython.display import Image, SVG

import matplotlib.pyplot as plt

from keras.utils import to_categorical

import os

from six.moves.urllib.request import urlretrieve

%matplotlib inline

import gzip

import numpy as np

import sys

import keras

from keras.datasets import mnist

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from keras import regularizers



try:

   import cPickle as pickle

except:

   import pickle



from sklearn import svm

from sklearn.metrics import classification_report,accuracy_score





print ('Started.....')



# Loads the training and test data sets (ignoring class labels)

#(x_train, _), (x_test, _) = mnist.load_data()





def extract_data(num_images):

    url = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'

    print (url)

    filepath, _ = urlretrieve(url)

    statinfo = os.stat(filepath)

    print('Succesfully downloaded', filepath, statinfo.st_size, 'bytes.')

    with gzip.open(filepath) as bytestream:

        bytestream.read(16)

        buf = bytestream.read(28 * 28 * num_images)

        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

        data = data.reshape(num_images, 28,28)

        os.remove (filepath)

    return data





f = open('../input/mnist.pkl/mnist.pkl', 'rb')

if sys.version_info < (3,):

    data = pickle.load(f)

else:

    data = pickle.load(f, encoding='bytes')

f.close()

(x_train, _), (x_test, _) = data















# Scales the training and test data to range between 0 and 1.

max_value = float(x_train.max())

x_train = x_train.astype('float32') / max_value

x_test = x_test.astype('float32') / max_value



x_train.shape, x_test.shape



x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))



(x_train.shape, x_test.shape)

# ((60000, 784), (10000, 784))







# input dimension = 784

input_dim = x_train.shape[1]

encoding_dim = 32



compression_factor = float(input_dim) / encoding_dim

print("Compression factor: %s" % compression_factor)



autoencoder = Sequential()

autoencoder.add(

    Dense(encoding_dim, input_shape=(input_dim,), activation='relu')

)

autoencoder.add(

    Dense(input_dim, activation='sigmoid')

)



autoencoder.summary()



input_img = Input(shape=(input_dim,))

encoder_layer = autoencoder.layers[0]

encoder = Model(input_img, encoder_layer(input_img))



encoder.summary()



epochs=50

batch_size=128

    

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder_train = autoencoder.fit(x_train, x_train,

                epochs,

                batch_size,

                shuffle=True,

                validation_data=(x_test, x_test))







num_images = 10

np.random.seed(42)

random_test_images = np.random.randint(x_test.shape[0], size=num_images)



encoded_imgs = encoder.predict(x_test)

decoded_imgs = autoencoder.predict(x_test)



plt.figure(figsize=(18, 4))



for i, image_idx in enumerate(random_test_images):

    # plot original image

    ax = plt.subplot(3, num_images, i + 1)

    plt.imshow(x_test[image_idx].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    # plot encoded image

    #ax = plt.subplot(3, num_images, num_images + i + 1)

    #plt.imshow(encoded_imgs[image_idx].reshape(8, 4))

    #plt.gray()

    #ax.get_xaxis().set_visible(False)

    #ax.get_yaxis().set_visible(False)



    # plot reconstructed image

    ax = plt.subplot(3, num_images, 2*num_images + i + 1)

    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()