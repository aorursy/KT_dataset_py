import numpy as np

import pickle

import cv2

from os import listdir

from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

traindir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train"

validdir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid"

testdir = "../input/new-plant-diseases-dataset/test/test"



train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   fill_mode='nearest')



valid_datagen = ImageDataGenerator(rescale=1./255)



batch_size = 128

training_set = train_datagen.flow_from_directory(traindir,

                                                 target_size=(224, 224),

                                                 batch_size=batch_size,

                                                 class_mode='categorical')



valid_set = valid_datagen.flow_from_directory(validdir,

                                            target_size=(224, 224),

                                            batch_size=batch_size,

                                            class_mode='categorical')

class_dict = training_set.class_indices

print(class_dict)
li = list(class_dict.keys())

print(li)
train_num = training_set.samples

valid_num = valid_set.samples

print("train_num is:",train_num)

print("valid_num is:",valid_num)
depth,height, width=3,224,224

n_classes=len(li)

print("n_classes:",n_classes)
# model = Sequential()

# inputShape = (height, width, depth)

# chanDim = -1



# model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

# model.add(Activation("relu"))

# model.add(BatchNormalization(axis=chanDim))

# model.add(MaxPooling2D(pool_size=(3, 3)))

# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding="same"))

# model.add(Activation("relu"))

# model.add(BatchNormalization(axis=chanDim))

# model.add(Conv2D(64, (3, 3), padding="same"))

# model.add(Activation("relu"))

# model.add(BatchNormalization(axis=chanDim))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

# model.add(Conv2D(128, (3, 3), padding="same"))

# model.add(Activation("relu"))

# model.add(BatchNormalization(axis=chanDim))

# model.add(Conv2D(128, (3, 3), padding="same"))

# model.add(Activation("relu"))

# model.add(BatchNormalization(axis=chanDim))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

# model.add(Flatten())

# model.add(Dense(1024))

# model.add(Activation("relu"))

# model.add(BatchNormalization())

# model.add(Dropout(0.5))

# model.add(Dense(n_classes))

# model.add(Activation("softmax"))

# model.summary()
# model.compile(optimizer='adam',

#               loss='categorical_crossentropy',

#               metrics=['accuracy'])
# #fitting images to CNN

# history = model.fit_generator(training_set,

#                          steps_per_epoch=train_num//batch_size,

#                          validation_data=valid_set,

#                          epochs=20,

#                          validation_steps=valid_num//batch_size,

#                          )
# move the model 

%cp -arvf "../input/cnn-model/cnn_model.h5" ./



from keras.callbacks import ModelCheckpoint

from keras.models import Sequential, load_model

# define the checkpoint

filepath = "./cnn_model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]



# load the model

model = load_model(filepath)

# fit the model

history = model.fit_generator(training_set,

                         steps_per_epoch=train_num//batch_size,

                         validation_data=valid_set,

                         epochs=30,

                         validation_steps=valid_num//batch_size,

                         callbacks=callbacks_list

                         )
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)



#accuracy plot

plt.plot(epochs, acc, color='green', label='Training Accuracy')

plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.figure()

#loss plot

plt.plot(epochs, loss, color='pink', label='Training Loss')

plt.plot(epochs, val_loss, color='red', label='Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()



plt.show()
x_test, y_test = valid_set.next() 

print("[INFO] Calculating model accuracy")

scores = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {scores[1]}")


# predicting an image

import os

import matplotlib.pyplot as plt

from keras.preprocessing import image

import numpy as np

directory="../input/new-plant-diseases-dataset/test/test"

files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))]

for i in range(0,10):

    image_path = files[i]

    new_img = image.load_img(image_path, target_size=(224, 224))

    img = image.img_to_array(new_img)

    img = np.expand_dims(img, axis=0)

    img = img/255

    prediction = model.predict(img)

    probabilty = prediction.flatten()

    max_prob = probabilty.max()

    index=prediction.argmax(axis=-1)[0]

    class_name = li[index]

    #ploting image with predicted class name        

    plt.figure(figsize = (4,4))

    plt.imshow(new_img)

    plt.axis('off')

    plt.title(class_name+" "+ str(max_prob)[0:4]+"%")

    plt.show()

        
#Confution Matrix and Classification Report

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict_generator(valid_set, valid_num//batch_size+1)
class_dict = valid_set.class_indices

li = list(class_dict.keys())

print(li)
y_pred = np.argmax(y_pred, axis=1)

print('Confusion Matrix')

print(confusion_matrix(valid_set.classes, y_pred))

print('Classification Report')

target_names =li ## ['Peach___Bacterial_spot', 'Grape___Black_rot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

print(classification_report(valid_set.classes, y_pred, target_names=target_names))
# save the model to disk

print("[INFO] Saving model...")

filepath="cnn_model.h5"

model.save(filepath)



# save the history to disk

print("[INFO] Saving history...")

pickle.dump(model,open('history.pkl', 'wb'))