from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K, models,layers
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf
adam = Adam(lr=1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)

import numpy as np
import pickle as pkl
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2 as cv
from skimage.io import imread, imshow, imread_collection, concatenate_images,imsave

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.utils.vis_utils import plot_model


import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
%matplotlib inline
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



def get_model():
    input_img = Input(shape=(80, 80, 3))  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoder = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (4, 4), activation='relu', padding='same')(encoder)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (1,1), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    return autoencoder

    





#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
print(get_model().summary())
plot_model(get_model(), to_file='model_plot.png', show_shapes=True, show_layer_names=True)



mc = ModelCheckpoint('non_mitotic_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
base_model = get_model()

folder = '../input/nuclei-images/'
x_train,x_test = pkl.load(open(folder+'mitotic.pkl','rb'))
nx_train,nx_test = pkl.load(open(folder+'non_mitotic.pkl','rb'))
print(nx_train.shape, x_train.shape)
x_train = np.concatenate((x_train, nx_train))
x_test = np.concatenate((x_test, nx_test))
print(x_train.shape, x_test.shape)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

autoencoder_train = base_model.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[mc])

base_model = load_model('non_mitotic_model.h5')

train_data = x_train

nm_len = len(nx_train)
m_len = len(x_train) - nm_len
train_labels = [1 for _ in range(m_len)]
train_labels.extend([0 for _ in range(nm_len)])

nm_len = len(nx_test)
m_len = len(x_test) - nm_len
test_labels = [1 for _ in range(m_len)]
test_labels.extend([0 for _ in range(nm_len)])

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_labels, num_classes=2)
test_Y_one_hot = to_categorical(test_labels, num_classes=2)

# Display the change for category label using one-hot encoding
print('Original label:', train_labels[-1])
print('After conversion to one-hot:', train_Y_one_hot[-1])

train_X,valid_X,train_label,valid_label = train_test_split(train_data,train_Y_one_hot,test_size=0.2,random_state=13)
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape
def get_encoder():
    input_img = Input(shape=(80, 80, 3))  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoder = MaxPooling2D((2, 2), padding='same')(x)
    return encoder
def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out
input_img = Input(shape=(80,80,3))
num_classes = 2
encode = get_encoder()
full_model = Model((input_img,fc(encode)))
print(full_model.layers)
for l1,l2 in zip(full_model.layers[0:7],base_model.layers[0:7]):
    l1.set_weights(l2.get_weights())
base_model.get_weights()[0][1]
full_model.get_weights()[0][1]

for layer in full_model.layers[0:7]:
    layer.trainable = False
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
full_model.summary()
classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=100,verbose=1,validation_data=(valid_X, valid_label))

mc = ModelCheckpoint('non_mitotic_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


nmitotic_model = get_model()

folder = '../input/nuclei-images/'
x_train,x_test = pkl.load(open(folder+'non_mitotic.pkl','rb'))
#nx_train,nx_test = pkl.load(open(folder+'non_mitotic.pkl','rb'))

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#nx_train = nx_train.astype('float32') / 255.
#nx_test = nx_test.astype('float32') / 255.

nmitotic_model.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[mc])
mc = ModelCheckpoint('mitotic_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

mitotic_model = get_model()
folder = '../input/nuclei-images/'
x_train,x_test = pkl.load(open(folder+'mitotic.pkl','rb'))
#nx_train,nx_test = pkl.load(open(folder+'non_mitotic.pkl','rb'))

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#nx_train = nx_train.astype('float32') / 255.
#nx_test = nx_test.astype('float32') / 255.
mitotic_model.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[mc])
mc = ModelCheckpoint('base_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

base_model = get_model()
folder = '../input/nuclei-images/'
x_train,x_test = pkl.load(open(folder+'mitotic.pkl','rb'))
nx_train,nx_test = pkl.load(open(folder+'non_mitotic.pkl','rb'))

x_train = np.append(x_train,nx_train,0)
x_test = np.append(x_test,nx_test,0)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#nx_test = nx_test.astype('float32') / 255.\
base_model.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[mc])
folder = '../input/nuclei-images/'
_,mx_test = pkl.load(open(folder+'mitotic.pkl','rb'))
_,nx_test = pkl.load(open(folder+'non_mitotic.pkl','rb'))

mx_test = mx_test.astype('float32') / 255.
nx_test = nx_test.astype('float32') / 255.
x_train.shape
from keras.models import load_model
base_model = load_model('base_model.h5')

decoded_imgs = base_model.predict(x_test)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
from keras.models import load_model
mitotic_model = load_model('mitotic_model.h5')

decoded_imgs = mitotic_model.predict(mx_test)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
n_mitotic_model = load_model('non_mitotic_model.h5')
decoded_imgs = nmitotic_model.predict(mx_test)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(mx_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
from keras.models import load_model
mitotic_model = load_model('mitotic_model.h5')
decoded_imgs = mitotic_model.predict(nx_test)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(nx_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
from keras.models import load_model
nmitotic_model = load_model('non_mitotic_model.h5')
decoded_imgs = nmitotic_model.predict(nx_test)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(nx_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
def euclidean(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=[1,2,3]))

def get_distances(d, x):
    y = []
    for i in range(len(d)):
        y.append(euclidean(d[i],x[i]))
    return y


num_classes = 2
pmodel = load_model('mitotic_model.h5')
nmodel = load_model('non_mitotic_model.h5')

mx_train,mx_test = pkl.load(open(folder+'mitotic.pkl','rb'))
nx_train,nx_test = pkl.load(open(folder+'non_mitotic.pkl','rb'))


x1 = pmodel.predict(mx_train)
x2 = nmodel.predict(mx_train)

x1 = get_distances(mx_train,x1)
x2 = get_distances(mx_train,x2)

x = [(x1[i],x2[i]) for i in range(len(x1))]
y = [1 for _ in range(len(x1)+len(x2))]


x1 = pmodel.predict(nx_train)
x2 = nmodel.predict(nx_train)

x1 = get_distances(nx_train,x1)
x2 = get_distances(nx_train,x2)

xb = [(x1[i],x2[i]) for i in range(len(x1))]
yb = [0 for _ in range(len(x1)+len(x2))]

x.extend(xb)
y.extend(yb)

x = np.asarray(x, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()
model.add(Dense(2, activation='relu',))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
a = [1,2,3,4].extend([2,3,4])
print(a)
train('../input/nuclei-images/')
tx = np.asarray([1,2,3])
tx = np.reshape(tx,(tx.shape[0],1))

ty = np.asarray([2,58,5])
ty = np.reshape(ty,(ty.shape[0],1))

np.append(tx,ty,axis=1)



%%time
X_train = np.zeros((len(train_ids)-86, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
missing_count = 0
print('Getting train images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_+''
    try:
        img = imread(path)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n-missing_count] = img
    except:
#         print(" Problem with: "+path)
        missing_count += 1

X_train = X_train.astype('float32') / 255.
print("Total missing: "+ str(missing_count))