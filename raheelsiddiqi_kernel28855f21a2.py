import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import os

import random

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.callbacks import ReduceLROnPlateau



%matplotlib inline
total_images_train_normal = os.listdir('../input/chest_xray/chest_xray/train/NORMAL')

total_images_train_pneumonia = os.listdir('../input/chest_xray/chest_xray/train/PNEUMONIA')
sample_normal = random.sample(total_images_train_normal,6)

f,ax = plt.subplots(2,3,figsize=(15,9))



for i in range(0,6):

    im = cv2.imread('../input/chest_xray/chest_xray/train/NORMAL/'+sample_normal[i])

    ax[i//3,i%3].imshow(im)

    ax[i//3,i%3].axis('off')

f.suptitle('Normal Lungs')

plt.show()
sample_pneumonia = random.sample(total_images_train_pneumonia,6)

f,ax = plt.subplots(2,3,figsize=(15,9))



for i in range(0,6):

    im = cv2.imread('../input/chest_xray/chest_xray/train/PNEUMONIA/'+sample_pneumonia[i])

    ax[i//3,i%3].imshow(im)

    ax[i//3,i%3].axis('off')

f.suptitle('Pneumonia Lungs')

plt.show()
sns.set_style('whitegrid')

sns.barplot(x=['Normal','Pneumonia'],y=[len(total_images_train_normal),len(total_images_train_pneumonia)])
image_height = 200

image_width = 200

batch_size = 10

no_of_epochs  = 50
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(image_height,image_width,3),activation='relu'))

model.add(Conv2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=128,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=15,

                                   shear_range=0.2,

                                   zoom_range=0.2

                                   )



test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',target_size=(image_width, image_height),batch_size=batch_size,class_mode='binary')



test_set = test_datagen.flow_from_directory('../input/chest_xray/chest_xray/test',target_size=(image_width, image_height),batch_size=batch_size,class_mode='binary')
history = model.fit_generator(training_set,

                    steps_per_epoch=5216//batch_size,

                    epochs=no_of_epochs,

                    validation_data=test_set,

                    validation_steps=624//batch_size

                   )
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

fig = plt.figure(figsize=(16,9))



plt.subplot(1, 2, 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()
no_steps = len(test_set)

result = model.evaluate_generator(test_set, steps=no_steps)

print("Test-set classification accuracy: {0:.2%}".format(result[1]))
# Preparing test data

import glob

from pathlib import Path

from keras.utils import to_categorical



normal_cases_dir = Path('../input/chest_xray/chest_xray/test/NORMAL')

pneumonia_cases_dir = Path('../input/chest_xray/chest_xray/test/PNEUMONIA')



normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



test_data = []

test_labels = []



for img in normal_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (image_width,image_height))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = [0]

    test_data.append(img)

    test_labels.append(label)

                      

for img in pneumonia_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (image_width,image_height))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = [1]

    test_data.append(img)

    test_labels.append(label)

    



test_data = np.array(test_data)

test_labels = np.array(test_labels)



print("Total number of test examples: ", test_data.shape)

print("Total number of labels:", test_labels.shape)
# Evaluation on test dataset

test_loss, test_score = model.evaluate(test_data, test_labels, batch_size=16)

print("Loss on test set: ", test_loss)

print("Accuracy on test set: ", test_score)
# Get predictions

preds = model.predict(test_data, batch_size=16,verbose=1)

preds=np.around(preds)

orig_test_labels=test_labels

print(preds.shape)

print(orig_test_labels.shape)
# Get the confusion matrix

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



cm  = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
# Calculate Precision and Recall

tn, fp, fn, tp = cm.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)

specificity=tn/(tn+fp)



print("Sensitivity (Recall) of the model is {:.2f}".format(recall))

print("Specificity of the model is {:.2f}".format(specificity))

print("Precision of the model is {:.2f}".format(precision))