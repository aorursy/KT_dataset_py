import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Reshape, Flatten



import random

import os

print(os.listdir("../input"))
os.listdir('../input/notmnist/notMNIST/notMNIST_small')
from PIL import Image

im = Image.open("../input/notmnist/notMNIST/notMNIST_small/A/MjAwcHJvb2Ztb29uc2hpbmUgcmVtaXgudHRm.png")

print(im.format, im.size, im.mode)
a_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/A')

b_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/B')

c_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/C')

d_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/D')

e_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/E')

f_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/F')

g_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/G')

h_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/H')

i_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/I')

j_dir = os.path.join('../input/notmnist/notMNIST/notMNIST_small/J')





print('Total training A images:', len(os.listdir(a_dir)))

print('Total training B images:', len(os.listdir(b_dir)))

print('Total training C images:', len(os.listdir(c_dir)))

print('Total training D images:', len(os.listdir(d_dir)))

print('Total training E images:', len(os.listdir(e_dir)))

print('Total training F images:', len(os.listdir(f_dir)))

print('Total training G images:', len(os.listdir(g_dir)))

print('Total training H images:', len(os.listdir(h_dir)))

print('Total training I images:', len(os.listdir(i_dir)))

print('Total training J images:', len(os.listdir(j_dir)))



a_files = os.listdir(a_dir)

print(a_files[:5])



b_files = os.listdir(b_dir)

print(b_files[:5])



c_files = os.listdir(c_dir)

print(c_files[:5])



d_files = os.listdir(d_dir)

print(d_files[:5])



e_files = os.listdir(e_dir)

print(e_files[:5])



f_files = os.listdir(f_dir)

print(f_files[:5])



g_files = os.listdir(g_dir)

print(g_files[:5])



h_files = os.listdir(h_dir)

print(h_files[:5])



i_files = os.listdir(i_dir)

print(i_files[:5])



j_files = os.listdir(j_dir)

print(j_files[:5])
pic_index = 2



next_a = [os.path.join(a_dir, fname) 

                for fname in a_files[pic_index-2:pic_index]]

next_b = [os.path.join(b_dir, fname) 

                for fname in b_files[pic_index-2:pic_index]]

next_c = [os.path.join(c_dir, fname) 

                for fname in c_files[pic_index-2:pic_index]]

next_d = [os.path.join(d_dir, fname) 

                for fname in d_files[pic_index-2:pic_index]]

next_e = [os.path.join(e_dir, fname) 

                for fname in e_files[pic_index-2:pic_index]]

next_f = [os.path.join(f_dir, fname) 

                for fname in f_files[pic_index-2:pic_index]]

next_g = [os.path.join(g_dir, fname) 

                for fname in g_files[pic_index-2:pic_index]]

next_h = [os.path.join(h_dir, fname) 

                for fname in h_files[pic_index-2:pic_index]]

next_i = [os.path.join(i_dir, fname) 

                for fname in i_files[pic_index-2:pic_index]]

next_j = [os.path.join(j_dir, fname) 

                for fname in j_files[pic_index-2:pic_index]]



for i, img_path in enumerate(next_a+next_b+next_c+next_d+next_e+next_f+next_g+next_h+next_i+next_j):

    img = mpimg.imread(img_path)

    plt.imshow(img)

    plt.axis('Off')

    plt.show()
X = []

labels = []

DATA_PATH = '../input/notmnist/notMNIST/notMNIST_small'

# for each folder (holding a different set of letters)

for directory in os.listdir(DATA_PATH):

    # for each image

    for image in os.listdir(DATA_PATH + '/' + directory):

        # open image and load array data

        try:

            file_path = DATA_PATH + '/' + directory + '/' + image

            img = Image.open(file_path)

            img.load()

            img_data = np.asarray(img, dtype=np.int16)

            # add image to dataset

            X.append(img_data)

            # add label to labels

            labels.append(directory)

        except:

            None # do nothing if couldn't load file

N = len(X) # number of images

img_size = len(X[0]) # width of image

# add our single channel for processing purposes

X = np.asarray(X).reshape(N, img_size, img_size,1) 

# convert to one-hot

labels = to_categorical(list(map(lambda x: ord(x)-ord('A'), labels)), 10) 
# TRAINING_DIR = "../input/notmnist/notMNIST/notMNIST_small/"

# training_datagen = ImageDataGenerator(

#     rescale = 1./255,

#     rotation_range=40,

#     width_shift_range=0.2,

#     height_shift_range=0.2,

#     shear_range=0.2,

#     zoom_range=0.2,

#     horizontal_flip=True,

#     fill_mode='nearest')



# train_generator = training_datagen.flow_from_directory(

#     TRAINING_DIR,

#     target_size=(28,28),

#     batch_size=32,

#     class_mode='categorical')
temp = list(zip(X, labels))

np.random.shuffle(temp)

X, labels = zip(*temp)

X, labels = np.asarray(X), np.asarray(labels)

PROP_TRAIN = 0.7 # proportion to use for training

NUM_TRAIN = int(N * PROP_TRAIN) # number to use for training

X_train, X_test = X[:NUM_TRAIN], X[NUM_TRAIN:]

labels_train, labels_test = labels[:NUM_TRAIN], labels[NUM_TRAIN:]
model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 28x28 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28,1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

#     tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution

#     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

#     tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# history = model.fit_generator(train_generator,epochs=25,verbose = 1)

csv_logger = tf.keras.callbacks.CSVLogger('training.csv', append=True)



# train model

model.fit(X_train, labels_train,

          epochs=40, batch_size=64,

          validation_data=[X_test, labels_test],

          callbacks=[csv_logger])



model.save("font_basic.h5")
data = np.genfromtxt('training.csv', delimiter=',')

data = data[1:][:,1:]



fig, axes = plt.subplots(1, 2)



# plot train and test accuracies

axes[0].plot(data[:,0]) # training accuracy

axes[0].plot(data[:,2]) # testing accuracy

axes[0].legend(['Training', 'Testing'])

axes[0].set_title('Accuracy Over Time')

axes[0].set_xlabel('epoch')

axes[0].set_ybound(0.0, 1.0)



# same plot zoomed into [0.85, 1.00]

axes[1].plot(np.log(1-data[:,0])) # training accuracy

axes[1].plot(np.log(1-data[:,2])) # testing accuracy

axes[1].legend(['Training', 'Testing'])

axes[1].set_title('Log-Inverse Accuracy')

axes[1].set_xlabel('epoch')

#axes[1].set_ybound(0.90,1.0)

plt.show()
score = model.evaluate(X_test, labels_test, verbose=False)

print('Loss: {}'.format(score[0]))

print('Accuracy: {}%'.format(np.round(10000*score[1])/100))