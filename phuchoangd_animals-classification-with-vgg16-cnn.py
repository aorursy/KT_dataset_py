import warnings

warnings.filterwarnings('ignore')

%config IPCompleter.greedy=True
import numpy as np                          # linear algebra

import os                                   # used for loading the data

from sklearn.metrics import confusion_matrix# confusion matrix to carry out error analysis

import seaborn as sn                        # heatmap

from sklearn.utils import shuffle           # shuffle the data

import matplotlib.pyplot as plt             # 2D plotting library

import cv2                                  # image processing library

import tensorflow as tf                     # best library ever
import os

print(os.listdir("/kaggle/input/inceptionv3"))
# Here's our 10 categories that we have to classify.

class_names = ['gallina', 'cane', 'cavallo', 'gatto', 'mucca', 'farfalla', 'elefante', 'ragno', 'scoiattolo', 'pecora']

class_names_label = {'gallina': 0,

                     'cane' : 1,

                     'cavallo' : 2,

                     'gatto' : 3,

                     'mucca' : 4,

                     'farfalla' : 5,

                     'elefante': 6,

                     'ragno': 7,

                     'scoiattolo': 8,

                     'pecora': 9

                    }

num_classes = len(class_names)

img_size = 128

num_channel = 3

animal_path = '/kaggle/input/animals10/animals/raw-img/'
def resize_to_square(image, size):

    h, w, c = image.shape

    ratio = size / max(h, w)

    resized_image = cv2.resize(image, (int(w*ratio), int(h*ratio)), cv2.INTER_AREA)

    return resized_image
def padding(image, min_height, min_width):

    h, w, c = image.shape



    if h < min_height:

        h_pad_top = int((min_height - h) / 2.0)

        h_pad_bottom = min_height - h - h_pad_top

    else:

        h_pad_top = 0

        h_pad_bottom = 0



    if w < min_width:

        w_pad_left = int((min_width - w) / 2.0)

        w_pad_right = min_width - w - w_pad_left

    else:

        w_pad_left = 0

        w_pad_right = 0

        

    return cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
def load_data(directory):

    """

        Load the data:

            - 14,034 images to train the network.

            - 10,000 images to evaluate how accurately the network learned to classify images.

    """

    output = []

    images = []

    labels = []

    file_names = []

    for folder in os.listdir(directory):

        curr_label = class_names_label[folder]

        for file in os.listdir(directory + "/" + folder):

            img_path = directory + "/" + folder + "/" + file

            curr_img = cv2.imread(img_path)

            curr_img = resize_to_square(curr_img, img_size)

            curr_img = padding(curr_img, img_size, img_size)

            images.append(curr_img)

            labels.append(curr_label)

            file_names.append(file)

    images, labels, file_names = shuffle(images, labels, file_names, random_state=817328462)     ### Shuffle the data !!!

    images = np.array(images, dtype = 'float32') ### Our images

    labels = np.array(labels, dtype = 'int32')   ### From 0 to num_classes-1!



    return images, labels, file_names
images, labels, file_names = load_data(animal_path)
x_data = images

y_data = labels
print(x_data.shape)

print(y_data.shape)
from sklearn.preprocessing import LabelEncoder



Labelencoder = LabelEncoder()

y_data = Labelencoder.fit_transform(y_data)

np.unique(y_data)
from keras.utils import to_categorical

num_classes = 10

Y = to_categorical(y_data,num_classes)

X = x_data
X_train = X[:20000]

y_train = Y[:20000]

X_temp = X[20001:]

y_temp = Y[20001:]



X_test = X_temp[:5000]

y_test = y_temp[:5000]

X_val = X_temp[5001:]

y_val = y_temp[5001:]
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

print(X_val.shape)

print(y_val.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Flatten

from tensorflow.keras.layers import Conv2D,MaxPool2D, GlobalAveragePooling2D

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.optimizers import SGD
TRAINING_NUMBER = len(y_train)

BATCH_SIZE = 128

EPOCHS = 10

LEARNING_RATE = 0.005

MOMENTUM = 0.9

DECAY = 1e-6
from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.applications.vgg16 import VGG16



image_input = Input(shape=(img_size,img_size,num_channel))

vgg_mod = VGG16(input_tensor=image_input, include_top=False,weights='imagenet')

vgg_mod.summary()
vgg_mod.trainable = False



for layer in vgg_mod.layers:

    if layer.name == 'block5_conv1':

        layer.trainable = True

    if layer.name == 'block4_conv1':

        layer.trainable = True    

    else:

        layer.trainable = False



vgg_mod.summary()
add_model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=vgg_mod.output_shape[1:]), # the nn will learn the good filter to use

    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])



model_vgg16 = Model(inputs=vgg_mod.input, outputs=add_model(vgg_mod.output))

model_vgg16.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],)

model_vgg16.summary()
from keras.callbacks import ReduceLROnPlateau



callbacks_list = [ReduceLROnPlateau(monitor='loss',factor=0.2,patience=3)]
hist_1=model_vgg16.fit(X_train,y_train,

                       batch_size=BATCH_SIZE,

                       epochs=EPOCHS, 

                       validation_data=(X_val, y_val), 

                       callbacks = callbacks_list, 

                       verbose=1)
model_vgg16.save_weights("model_vgg16.h5")

#loaded_model.load_weights("model_vgg16.h5"
plt.plot(hist_1.history['loss'])

plt.plot(hist_1.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'val'])

plt.show()
plt.plot(hist_1.history['accuracy'])

plt.plot(hist_1.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'val'])

plt.show()
#Testing result

score = model_vgg16.evaluate(X_test, y_test, verbose=0)



print('Test loss:', score[0])

print('Test accuracy:', score[1])
#from keras.preprocessing.image import ImageDataGenerator

#from keras.callbacks import ModelCheckpoint



#train_datagen = ImageDataGenerator(

#        rotation_range=30, 

#        width_shift_range=0.1,

#        height_shift_range=0.1, 

#        horizontal_flip=True)

#train_datagen.fit(X_train)
#history_vgg16 = model_vgg16.fit_generator(

#    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),

#    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,

#    epochs=EPOCHS,

#    validation_data=(X_val, y_val),

#    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]

#)