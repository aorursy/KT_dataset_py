%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib as mpl
import keras

from keras.layers import Input, Dense, Flatten, Dropout

from keras.layers.normalization import BatchNormalization

from keras.models import Model

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator
import os, fnmatch

from skimage import io, transform

import numpy as np

from tqdm import tqdm

import pandas as pd

import shutil
vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights=None,

                                       # use weights='imagenet'

                                       input_tensor=None, input_shape=(224,224,3))

vgg16_full = keras.applications.vgg16.VGG16(include_top=True, weights=None, 

                                            # use weights='imagenet'

                                            input_tensor=None, input_shape=(224,224,3))
fc1_layer = vgg16_full.get_layer("fc1")

fc1_layer
fc2_layer = vgg16_full.get_layer("fc2")

fc2_layer
!ls ../input
labels_csv = pd.read_csv("../input/labels.csv")
breeds = pd.Series(labels_csv['breed'])

filenames = pd.Series(labels_csv['id'])
unique_breeds = np.unique(breeds)

labels = []

for breed in breeds:

    i = np.where(unique_breeds == breed)[0][0]

    labels.append(i)



n_breeds = np.max(labels) + 1

labels = np.eye(n_breeds)[labels]
filenames_train = []

filenames_validate = []



# move to validate folder

for i in tqdm(range(len(filenames))):

    label = unique_breeds[np.where(labels[i]==1.)][0]

    filename = '{}.jpg'.format(filenames[i])



    if i < 8000:

        new_dir = './sorted/train/{}/'.format(label)

        filenames_train.append(new_dir + filename)

    else:

        new_dir = './sorted/validate/{}/'.format(label)

        filenames_validate.append(new_dir + filename)

        

    if not os.path.exists(new_dir):

        os.makedirs(new_dir)

    

    shutil.copy("../input/train/{}.jpg".format(filenames[i]), new_dir + filename)
indices_train = np.argsort(filenames_train)

indices_val = np.argsort(filenames_validate)



sorted_filenames_train = np.array(filenames_train)[indices_train]

sorted_filenames_validate = np.array(filenames_validate)[indices_val]

sorted_labels_train = np.array(labels)[0:8000][indices_train]

sorted_labels_validate = np.array(labels)[8000:][indices_val]
print(unique_breeds[np.where(sorted_labels_train[50] == 1.)])

# should be equal to:

print(sorted_filenames_train[50])
def preprocess(img):

    input_img = preprocess_input(np.expand_dims(img, axis=0))

    return input_img[0]



train_datagen = ImageDataGenerator(preprocessing_function=preprocess)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess)
batch_size = 64



train_gen = train_datagen.flow_from_directory("./sorted/train", 

                                              batch_size=batch_size, 

                                              target_size=(224, 224), 

                                              shuffle=False)



val_gen = val_datagen.flow_from_directory("./sorted/validate", 

                                          batch_size=batch_size, 

                                          target_size=(224, 224), 

                                          shuffle=False)
x_train = vgg16.predict_generator(train_gen, 

                                  # steps=8000 // batch_size, 

                                  steps=1, 

                                  verbose=1)
x_val = vgg16.predict_generator(val_gen, 

                                # steps=2222 // batch_size, 

                                steps=1,

                                verbose=1)
y_train = sorted_labels_train[0:len(x_train)]

y_val = sorted_labels_validate[0:len(x_val)]
inputs = Input(shape=(7,7,512))



# Turn off training vgg16

for layer in vgg16.layers:

    layer.trainable = False

fc1_layer.trainable = False



x = Flatten()(inputs)

x = fc1_layer(x)

x = BatchNormalization()(x)

x = Dropout(0.8)(x)

x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.8)(x)

x = Dense(120, activation='softmax')(x)



model = Model(inputs=inputs, outputs=x)

model.summary()
model.compile(optimizer=keras.optimizers.Adam(), 

              loss=keras.losses.categorical_crossentropy, 

              metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=30, verbose=1, 

                    validation_data=(x_val, y_val))
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()