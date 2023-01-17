import numpy as np

import pandas as pd 

import os

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten

from keras.optimizers import RMSprop,SGD, Adam

from keras import regularizers

from tensorflow.keras.applications import MobileNetV2

import matplotlib.pyplot as plt

from keras.utils import plot_model
train_fake = '../input/real-and-fake-face-detection/real_and_fake_face/training_fake/'

train_real = '../input/real-and-fake-face-detection/real_and_fake_face/training_real/'



def plot_image(path, title):

    plt.figure(figsize=(10,10))

    for i in range(9):

        img = load_img(path +'/' +os.listdir(path)[i])

        plt.subplot(3,3,i+1)

        plt.imshow(img)

        if title=='Fake Faces':

            plt.title(os.listdir(path)[i][:4])

        plt.suptitle(title)

        plt.axis('off')

    return plt

plot_image(train_real, 'Real Faces').show()
plot_image(train_fake, 'Fake Faces').show()
path_data = '../input/real-and-fake-face-detection/real_and_fake_face/'

data_gen = ImageDataGenerator(rescale=1./255,

                              horizontal_flip=True,

                              zoom_range=0.2,

                              rotation_range=20,

                              shear_range=0.3,

                              width_shift_range=0.2,

                             )

training_set = data_gen.flow_from_directory(path_data,

                                            class_mode='binary',

                                            shuffle=True,

                                            target_size=(96,96),

                                            batch_size=64,

                                           )



training_set.class_indices


mobilenet = MobileNetV2(input_shape=(96, 96, 3),

                        include_top=False,

                        weights='imagenet'

                       )

model = tf.keras.models.Sequential([mobilenet,

                                    GlobalAveragePooling2D(),

                                    Dense(512, activation='relu'),

                                    BatchNormalization(),

                                    Dropout(0.3),

                                    Dense(1, activation='sigmoid')

                                   ])

model.compile(optimizer=Adam(lr=0.001),

              loss='binary_crossentropy',

              metrics=['accuracy']

             )

model.summary()
hist = model.fit(x=training_set, epochs=20)
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

plt.plot(hist.history['accuracy'])

plt.title('Accuracy vs Epoch')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()



plt.subplot(1,2,2)

plt.plot(hist.history['loss'])

plt.title('Loss vs Epoch')

plt.xlabel('Accuracy')

plt.ylabel('Loss')

plt.legend()

plt.show()
y_pred = model.predict(training_set)

y_pred = (y_pred < 0.5).astype(np.int)



from sklearn.metrics import classification_report, confusion_matrix

cm_test = confusion_matrix(training_set.classes, y_pred)

print('Confusion Matrix')

print(cm_test)



print('Classification Report')

print(classification_report(training_set.classes, y_pred, target_names=['fake', 'real']))



plt.figure(figsize=(6,6))

plt.imshow(cm_test, interpolation='nearest')

plt.colorbar()

tick_mark = np.arange(len(target_names))

_ = plt.xticks(tick_mark, ['fake', 'real'], rotation=90)

_ = plt.yticks(tick_mark, ['fake', 'real'])
model.save('spoofnet.h5')