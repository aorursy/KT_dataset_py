# Importing the Keras libraries and packages

import tensorflow as tf

from keras.applications.vgg16 import VGG16

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg19 import VGG19

from keras.models import Model

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers import Input, Lambda ,Dense ,Flatten

import numpy as np
#Initialising vgg16 and vgg 19

classifier_vgg16 = VGG16(input_shape= (224,224,3),include_top=False,weights='imagenet')



classifier_vgg16.summary()
#don't train existing weights

for layer in classifier_vgg16.layers:

    layer.trainable = False
classifier1 = classifier_vgg16.output#head mode

classifier1 = Flatten()(classifier1)#adding layer of flatten

classifier1 = Dense(units=1, activation='sigmoid')(classifier1)#again adding another layer of dense



model = Model(inputs = classifier_vgg16.input , outputs = classifier1)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Part 2 - Fitting the CNN to the images



from tensorflow.keras.preprocessing.image import ImageDataGenerator

#use the image data generator to import the images from the dataset

#data augmentation

train_datagen = ImageDataGenerator(rescale=1. / 255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1. / 255)
#makes sure you provide the same target as initialised for the image size

training_set = train_datagen.flow_from_directory('/kaggle/input/Data/train',

                                                 target_size=(224, 224),

                                                 batch_size=32,

                                                 class_mode='binary')



test_set = test_datagen.flow_from_directory('/kaggle/input/Data/test',

                                            target_size=(224, 224),

                                            batch_size=32,

                                            class_mode='binary')
#fit the model

#it will take some time to train

history = model.fit_generator(training_set,

                              validation_data=test_set,

                              epochs=5,

                              steps_per_epoch=len(training_set),

                              validation_steps=len(test_set))
#save the model

model.save('my_model_vgg16.h5')
#evaluate the model

loaded_model = tf.keras.models.load_model('my_model_vgg16.h5')

loaded_model.evaluate(test_set)
#accuarcy

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# for only one prediction

import numpy as np

from keras.preprocessing import image



test_image = image.load_img('/kaggle/input/Data/test/Covid/covid-19-pneumonia-28.png',target_size=(224,224))

plt.imshow(test_image)

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

result = loaded_model.predict(test_image)

training_set.class_indices

if result[0][0] == 1:

    prediction = 'Normal'

else:

    prediction = 'Covid'

print(prediction)
# for only one prediction

import numpy as np

from keras.preprocessing import image



test_image = image.load_img('/kaggle/input/Data/original test set/NORMAL2-IM-0112-0001.jpeg',target_size=(224,224))

plt.imshow(test_image)

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

result = loaded_model.predict(test_image)

training_set.class_indices

if result[0][0] == 1:

    prediction = 'Normal'

else:

    prediction = 'Covid'

print(prediction)
# plot confusion metrix

y_pred = []

y_test = []

import os



for i in os.listdir("/kaggle/input/Data/test/Normal"):

    img = image.load_img("/kaggle/input/Data/test/Normal/" + i, target_size=(224,224))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    p = model.predict(img)

    y_test.append(p[0, 0])

    y_pred.append(1)



for i in os.listdir("/kaggle/input/Data/test/Covid"):

    img = image.load_img("/kaggle/input/Data/test/Covid/" + i, target_size=(224,224))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    p = model.predict(img)

    y_test.append(p[0, 0])

    y_pred.append(0)



y_pred = np.array(y_pred)

y_test = np.array(y_test).astype(int)





from sklearn.metrics import confusion_matrix

import seaborn as sns

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, cmap="plasma", annot=True)



from sklearn.metrics import classification_report

print(classification_report(y_pred, y_test))
#Initialising vgg16 and vgg 19

classifier_vgg19 = VGG19(input_shape= (224,224,3),include_top=False,weights='imagenet')



classifier_vgg19.summary()
#don't train existing weights

for layer in classifier_vgg19.layers:

    layer.trainable = False
classifier2 = classifier_vgg19.output#head mode

classifier2 = Flatten()(classifier2)#adding layer of flatten

classifier2 = Dense(units=64, activation='relu')(classifier2)#adding layer of dense

classifier2 = Dense(units=1, activation='sigmoid')(classifier2)#again adding another layer of dense



model1 = Model(inputs = classifier_vgg19.input , outputs = classifier2)

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model1.summary()
# Part 2 - Fitting the CNN to the images



from tensorflow.keras.preprocessing.image import ImageDataGenerator

#use the image data generator to import the images from the dataset

#data augmentation

train_datagen = ImageDataGenerator(rescale=1. / 255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1. / 255)
#makes sure you provide the same target as initialised for the image size

training_set = train_datagen.flow_from_directory('/kaggle/input/Data/train',

                                                 target_size=(224, 224),

                                                 batch_size=32,

                                                 class_mode='binary')



test_set = test_datagen.flow_from_directory('/kaggle/input/Data/test',

                                            target_size=(224, 224),

                                            batch_size=32,

                                            class_mode='binary')
#fit the model

#it will take some time to train

history = model1.fit_generator(training_set,

                              validation_data=test_set,

                              epochs=10,

                              steps_per_epoch=len(training_set),

                              validation_steps=len(test_set))
#save the model

model1.save('my_model_vgg19.h5')
#evaluate the model

loaded_model1 = tf.keras.models.load_model('my_model_vgg19.h5')

loaded_model1.evaluate(test_set)
#accuarcy

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# for only one prediction

import numpy as np

from keras.preprocessing import image



test_image = image.load_img('/kaggle/input/Data/train/Covid/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-003-fig4b.png',target_size=(224,224))

plt.imshow(test_image)

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

result = loaded_model1.predict(test_image)

training_set.class_indices

if result[0][0] == 1:

    prediction = 'Normal'

else:

    prediction = 'Covid'

print(prediction)
# for only one prediction

import numpy as np

from keras.preprocessing import image



test_image = image.load_img('/kaggle/input/Data/original test set/NORMAL2-IM-0112-0001.jpeg',target_size=(224,224))

plt.imshow(test_image)

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

result = loaded_model1.predict(test_image)

training_set.class_indices

if result[0][0] == 1:

    prediction = 'Normal'

else:

    prediction = 'Covid'

print(prediction)
# plot confusion metrix

y_pred2 = []

y_test = []

import os



for i in os.listdir("/kaggle/input/Data/test/Normal"):

    img = image.load_img("/kaggle/input/Data/test/Normal/" + i, target_size=(224,224))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    p = model1.predict(img)

    y_test.append(p[0, 0])

    y_pred2.append(1)



for i in os.listdir("/kaggle/input/Data/test/Covid"):

    img = image.load_img("/kaggle/input/Data/test/Covid/" + i, target_size=(224,224))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    p = model1.predict(img)

    y_test.append(p[0, 0])

    y_pred2.append(0)



y_pred2 = np.array(y_pred2)

y_test = np.array(y_test).astype(int)
from sklearn.metrics import confusion_matrix

import seaborn as sns

cm = confusion_matrix(y_pred2, y_test)

sns.heatmap(cm, cmap="plasma", annot=True)



from sklearn.metrics import classification_report

print(classification_report(y_pred2, y_test))