from keras.datasets import cifar10
import numpy as np
(X_train, _), (X_test, _) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
#X_train = X_train.reshape(X_train.shape[0], -1)
#X_test = X_test.reshape(X_test.shape[0], -1)
X_train.shape
from keras.layers import Input, Dense, UpSampling2D
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D


#input_img = Input(shape=(32, 32, 3))
#encoded = Dense(128, name='e1', activation='relu')(input_img)
#encoded = Dense(64, name='e2', activation='relu')(encoded)
#encoded = Dense(32, name='e3', activation='relu')(encoded)

#decoded = Dense(64, name='d1', activation='relu')(encoded)
#decoded = Dense(128, name='d2', activation='relu')(decoded)
#decoded = Dense(784, name='d3', activation='sigmoid')(decoded)

#autoencoder = Model(input_img, decoded)
#autoencoder.summary()

input_img = Input(shape=(32, 32, 3))
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D(pool_size=(2, 2))(encoded)

decoded = Conv2D(64, (3, 3), activation='relu', padding='same', name='d1')(encoded)
decoded = Conv2D(64, (3, 3), activation='relu', padding='same', name='d1')(encoded)
decoded = UpSampling2D(size=(2, 2), name='d2')(decoded)
decoded = Conv2D(32, (3, 3), activation='relu', padding='same', name='d3')(decoded)
decoded = UpSampling2D(size=(2, 2), name='d4')(decoded)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='d5')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
encoder = Model(input_img, encoded)
encoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

encoded_imgs = encoder.predict(X_test)

%matplotlib inline
import matplotlib.pyplot as plt


plt.imshow(X_test[25])
plt.gray()
plt.show()
img_to_find = encoded_imgs[25]
def custom_cosine_sim(a,b):
    return np.dot(a, b) / ( np.linalg.norm(a) * np.linalg.norm(b))
from scipy import spatial
cosine_list = []
for index_image,xt in enumerate(encoded_imgs):

    result = 1 - spatial.distance.cosine(img_to_find.reshape(-1), xt.reshape(-1))
    cosine_list.append(dict({'res':result, 'i':index_image}))
from operator import itemgetter
cosine_list.sort(key=itemgetter('res'), reverse=True)

cosine_list
%matplotlib inline
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=10,figsize=(20, 4))
plt.gray()
for indice, row in enumerate(ax):
    print (cosine_list[indice]['i'])
    row.imshow(X_test[cosine_list[indice]['i']])

plt.show()

