%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



plt.rcParams['font.size'] = 8

plt.rcParams['figure.figsize'] = (8,8)



import os

import numpy as np

import cv2

import pickle



from keras.models import Sequential, load_model

from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D

from keras.preprocessing.image import load_img, img_to_array

from keras import backend as K



from sklearn.cluster import KMeans
!find ../input/train -name "*.jpg" > train_images.txt



with open("train_images.txt", "r") as f:

    train_images = f.read().split('\n')[:-1]



img_width, img_height, channels = 224, 224, 3

input_shape = (img_width, img_height, channels)



def load_data(files):

    arr = np.empty((len(files), img_width, img_height, channels), dtype=np.float32)

    for i, imgfile in enumerate(files):

        img = load_img(imgfile)

        x = img_to_array(img).reshape(img_width, img_height, channels)

        x = x.astype('float32') / 255.

        arr[i] = x

    return arr



X = load_data(files=train_images)
nb_rows, nb_cols = 5, 5

plt.figure(figsize=(15,15))

for k in range(nb_rows * nb_cols):

    plt.subplot(nb_rows, nb_cols, k+1)

    plt.imshow(X[k])

    plt.axis('off')
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(UpSampling2D((2, 2)))

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(UpSampling2D((2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(UpSampling2D((2, 2)))

model.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))



model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])



model.summary()
model.fit(X, X, epochs=20, batch_size=128, shuffle=True)
def plot_some(im_list):

    plt.figure(figsize=(15,4))

    for i, array in enumerate(im_list):

        plt.subplot(1, len(im_list), i+1)

        plt.imshow(array)

        plt.axis('off')

    plt.show()



img_decoded = model.predict(X[:5])



print('Before autoencoding:')

plot_some(X[:5])

print('After decoding:')

plot_some(img_decoded)
X_sample = X[:100]

print(X_sample.shape)



get_encoded = K.function([model.layers[0].input], [model.layers[5].output])

encoded_sample = get_encoded([X_sample])[0]

print(encoded_sample.shape)
for n_image in range(0, 5):

    

    plt.figure(figsize=(12,4))



    plt.subplot(1,4,1)

    plt.imshow(X_sample[n_image][:,:,::-1])

    plt.axis('off')

    plt.title('Original Image')



    plt.subplot(1,4,2)

    plt.imshow(encoded_sample[n_image].mean(axis=-1))

    plt.axis('off')

    plt.title('Encoded Mean')



    plt.subplot(1,4,3)

    plt.imshow(encoded_sample[n_image].max(axis=-1))

    plt.axis('off')

    plt.title('Encoded Max')



    plt.subplot(1,4,4)

    plt.imshow(encoded_sample[n_image].std(axis=-1))

    plt.axis('off')

    plt.title('Encoded Std')



    plt.show()
X_encoded = np.empty((len(X), 28, 28, 8), dtype='float32')



step = 100

for i in range(0, len(X), step):

    x_batch = get_encoded([X[i:i+step]])[0]

    X_encoded[i:i+step] = x_batch



print(X_encoded.shape)
X_encoded_reshape = X_encoded.reshape(X_encoded.shape[0], X_encoded.shape[1]*X_encoded.shape[2]*X_encoded.shape[3])

print(X_encoded_reshape.shape)
n_clusters = 100                      # just for fun



km = KMeans(n_clusters=n_clusters)

km.fit(X_encoded_reshape)
plt.figure(figsize=(20, 5))

cluster_elements = [(km.labels_==i).sum() for i in range(n_clusters)]

plt.bar(range(n_clusters), cluster_elements, width=1)
average_clusters_encoded = []

for i in range(n_clusters):

    average_clusters_encoded.append(X_encoded[km.labels_==i].mean(axis=0))



average_clusters_encoded = np.asarray(average_clusters_encoded)



print(average_clusters_encoded.shape)
get_decoded = K.function([model.layers[6].input],

                         [model.layers[-1].output])



decoded_clusters = get_decoded([average_clusters_encoded])
plt.figure(figsize=(20, 20))



for i in range(n_clusters):

    plt.subplot(10, 10, i+1)

    plt.imshow(decoded_clusters[0][i][:,:,::-1])

    plt.title('Cluster {}'.format(i))

    plt.axis('off')



plt.show()
plt.figure(figsize=(20, 20))



cluster = 5

rows, cols = 10, 10

start = 0



labels = np.where(km.labels_==cluster)[0][start:start+rows*cols]

for i, label in enumerate(labels):

    plt.subplot(rows, cols, i+1)

    plt.imshow(X[label])

    plt.axis('off')