import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import random
from sklearn.datasets import fetch_olivetti_faces
dataset = fetch_olivetti_faces()
dataset
print(dir(dataset))
print(dataset['DESCR'])
print(len(dataset['images']))
print(len(dataset['target']))
fig = plt.figure(figsize=(20,20))
for x in range(400):
    plt.subplot(20,20,x+1)
    plt.imshow(dataset['images'][x])
plt.show()   
noisy = dataset['images'] + dataset['images'].std()*3*np.random.random(64)
plt.imshow(noisy[0])
plt.imshow(dataset['images'][0])
train = dataset['images'][:320]
test =  dataset['images'][320:400]
noisy_train = noisy[:320]
noisy_test = noisy[320:400]
train = train.reshape(320,64,64,1)
test = test.reshape(80,64,64,1)
noisy_train = noisy_train.reshape(320,64,64,1)
noisy_test = noisy_test.reshape(80,64,64,1)
input_img = Input(shape=(64, 64, 1))

nn = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
nn = MaxPooling2D((2, 2), padding='same')(nn)
nn = Conv2D(64, (3, 3), activation='relu', padding='same')(nn)
encoded = MaxPooling2D((2, 2), padding='same')(nn)


nn = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
nn = UpSampling2D((2, 2))(nn)
nn = Conv2D(64, (3, 3), activation='relu', padding='same')(nn)
nn = UpSampling2D((2, 2))(nn)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(nn)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
autoencoder.fit(noisy_train, train,
                epochs=1000,
                validation_data=(noisy_test, test))
decoded_imgs = autoencoder.predict(noisy_test)
plt.imshow(noisy_test[0].reshape(64,64))
plt.imshow(decoded_imgs[0].reshape(64,64))
plt.imshow(decoded_imgs[0].reshape(64,64))
plt.imshow(test[0].reshape(64,64))
