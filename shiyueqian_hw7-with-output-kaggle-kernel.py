import numpy as np 



import glob



files = glob.glob("/kaggle/input/homework-7/Book Files/books/*")

files
import os

dirs = '/kaggle/working/Book Files/'

if not os.path.exists(dirs):

    os.makedirs(dirs)

with open("/kaggle/working/Book Files/corpus.txt", "w") as outfile:

    for file in files:

        with open(file, "r",encoding="ascii",errors="ignore") as infile:

            outfile.write(infile.read())
import re

import string

raw_text = open("/kaggle/working/Book Files/corpus.txt","r").read()

raw_text = raw_text.lower() 

raw_text = raw_text.translate(str.maketrans('', '', string.punctuation))

raw_text = re.sub(r"\W+"," ", raw_text)
raw_text
chars = sorted(list(set(raw_text)))

ord_values = [ord(character) for character in chars]

ord_rescale = [value/max(ord_values) for value in ord_values]

char_to_ascii = dict(zip(chars,ord_rescale))

char_to_int = dict((c, i) for i, c in enumerate(chars))

char_to_ascii
char_to_int
n_chars = len(raw_text)

n_vocab = len(chars)

print("Total Characters: %s" % n_chars)

print("Total Vocab: %s" % n_vocab)
win_size = 100
dataX = []

dataY=[]

for i in range(0, n_chars - win_size + 1, 1):

    seq_in = raw_text[i:i + win_size-1]

    seq_out = raw_text[i + win_size-1]

    dataX.append([char_to_ascii[char] for char in seq_in])

    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)

print("Total Patterns: %d" % n_patterns)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import LSTM

from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils
# reshape X to be [samples, time steps, features]

X = np.reshape(dataX, (n_patterns, win_size - 1, 1))

dataX = []

# one hot encode the output variable

y = np_utils.to_categorical(dataY)
print(X.shape)

print(y.shape)
import tensorflow as tf

# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():

    model = Sequential()

    model.add(LSTM(y.shape[1], input_shape=(X.shape[1], X.shape[2])))

    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint

filepath="/kaggle/working/Book Files/LSTM-{epoch:02d}-{loss:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]



# train model normally

model.fit(X, y, epochs=30, batch_size=256, callbacks=callbacks_list)
paths = sorted(glob.glob("/kaggle/working/Book Files/*hdf5"))

filename = paths[-1]

filename
model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer='adam')
text = "There are those who take mental phenomena naively, just as they would physical phenomena. This school of psychologists tends not to emphasize the object."

text = text.lower() 

text = text.translate(str.maketrans('', '', string.punctuation))

text = re.sub(r"\W+"," ", text)



pattern = []

for char in text:

    pattern.append(char_to_ascii[char])

int_to_char = dict((i, c) for i, c in enumerate(chars))

ascii_to_char = dict(zip(ord_rescale,chars))
count = 0  

res_text = ""

for i in range(1000): 

    x = np.reshape(pattern[:win_size-1], (1, win_size-1, 1))

    prediction = model.predict(x, verbose=0)

    index = np.argmax(prediction[0])

    result = int_to_char[index]

    res_text += result

    pattern.append(char_to_ascii[int_to_char[index]])

    pattern = pattern[1:len(pattern)]
res_text
import numpy as np 

import glob



filepath = sorted(glob.glob("/kaggle/input/homework-7/cifar-10-python/cifar-10-batches-py/*_batch*"))

filepath
def unpickle(file):

    import pickle

    with open(file, 'rb') as fo:

        dict = pickle.load(fo, encoding='bytes')

    return dict



for i in range(len(filepath)):

    batch = unpickle(filepath[i])

    if i == 0:

        X = batch[b'data']

        y = batch[b"labels"]

    else:

        X = np.append(X,batch[b'data'], axis=0)

        y = np.append(y,batch[b"labels"], axis=0)

print(X.shape)

print(y.shape)
bird_idx = np.argwhere(y == 2)

X_bird = np.squeeze(X[bird_idx])

y_bird = np.squeeze(y[bird_idx])

print(X_bird.shape)

print(y_bird.shape)
features = X_bird.reshape((len(X_bird), 3, 32, 32)).transpose(0, 2, 3, 1)

rgb = []

for i in range(6000):

    for j in range(32):

        for k in range(32):

            rgb.append(list(features[i][j][k]))

print(len(rgb))
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=4, random_state=0).fit(rgb)

centers = kmeans.cluster_centers_

centers
pixel = np.array(rgb).astype(float)

for label in range(4):

    idx = np.argwhere(kmeans.labels_ == label)

    pixel[idx] = centers[label]
from skimage.color import rgb2gray



gray = rgb2gray(features).reshape(6000,32,32,1)

print(gray.shape)
# split training set and test set

test_num = len(np.argwhere(np.array(batch[b"labels"]) == 2))

gray_train = gray[:6000-test_num]

gray_test = gray[6000-test_num:]
# change label to one hot

from keras.utils import np_utils



y_label = np_utils.to_categorical(kmeans.labels_).reshape(6000,32,32,4)

label_train = y_label[:(6000-test_num)]

label_test = y_label[(6000-test_num):]
import keras

from keras.models import Sequential,Input,Model

from keras.layers import Dense, Flatten,Softmax,Reshape

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint
batch_size = 128

epochs = 30

num_classes = 4

# set checkpoints

import os

dirs = '/kaggle/working/cifar-10/'

if not os.path.exists(dirs):

    os.makedirs(dirs)

filepath="/kaggle/working/cifar-10/color-{epoch:02d}-{loss:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')

callbacks_list = [checkpoint]
color_model = Sequential()

color_model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=(32,32,1),padding='same'))

color_model.add(MaxPooling2D((2, 2),padding='same'))

color_model.add(Conv2D(32, kernel_size=(5, 5), activation="relu",padding='same'))

color_model.add(MaxPooling2D((2, 2),padding='same'))

color_model.add(Flatten())

color_model.add(Dense(32, activation="relu"))

color_model.add(Dense(4096, activation="relu"))

color_model.add(Reshape((32,32,4)))

color_model.add(Dense(num_classes, activation='softmax'))

color_model.compile(loss="categorical_crossentropy", optimizer='adam')
model_train = color_model.fit(gray_train,label_train, batch_size=batch_size,epochs=epochs,callbacks=callbacks_list)
# achieve training error and test error of each epochs

filenames = sorted(glob.glob("/kaggle/working/cifar-10/color*.hdf5"))

train_err = []

test_err = []

for file in filenames:

    color_model.load_weights(file)

    color_model.compile(loss='categorical_crossentropy', optimizer='adam')

    train_err.append(color_model.evaluate(gray_train,label_train))

    test_err.append(color_model.evaluate(gray_test,label_test))
import matplotlib.pyplot as plt



plt.plot(range(30),train_err)

plt.plot(range(30),test_err)

plt.legend(["training error","tets error"])
# get the pixel array for original images, images after kmeans and colorized images

filename = filenames[-1]

color_model.load_weights(filename)

color_model.compile(loss='categorical_crossentropy', optimizer='adam')

img_input = gray_test[:10]

img_pred = color_model.predict(img_input)

img = features[6000-test_num:][:10]

img_kmeans = np.uint8(pixel.reshape(6000,32,32,3)[6000-test_num:][:10])

fig=plt.figure(figsize=(10, 30))

for i in range(10):

    img_output = np.uint8(np.array([centers[np.argmax(img)] for img in img_pred[i].reshape(-1,4)]).reshape(32,32,3))

    fig.add_subplot(10, 3, 3*i+1)

    plt.imshow(img[i])

    plt.axis('off')

    fig.add_subplot(10, 3, 3*i+2)

    plt.imshow(img_kmeans[i])

    plt.axis('off')

    fig.add_subplot(10, 3, 3*i+3)

    plt.imshow(img_output)

    plt.axis('off')

plt.show()

# the three columns represent original images, images after kmeans and colorized images