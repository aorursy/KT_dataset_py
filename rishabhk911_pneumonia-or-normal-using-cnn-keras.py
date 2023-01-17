# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print('Numer of Normal images in training data')
y=len(os.listdir("../input/chest_xray/chest_xray/train/NORMAL"))
# y-1 because one file is .DS_STORE
print(y-1)
print('Numer of Pneumonia Diagnosed images in training data')
x=len(os.listdir("../input/chest_xray/chest_xray/train/PNEUMONIA"))
# x-1 because one file is .DS_STORE
print(x-1)
from PIL import Image 
  
# Read image 
img = Image.open("../input/chest_xray/chest_xray/train/NORMAL/"+'IM-0115-0001.jpeg') 
plt.title('Normal Image')
plt.imshow(img)
img2=Image.open("../input/chest_xray/chest_xray/train/PNEUMONIA/"+'person1_bacteria_1.jpeg') 

plt.title('Pneumonia Image')
plt.imshow(img2)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '../input/chest_xray/chest_xray/train',
        target_size=(128, 128),
        batch_size=16,
        class_mode='binary')

test_gen = test_datagen.flow_from_directory(
        '../input/chest_xray/chest_xray/test',
        target_size=(128, 128),
        batch_size=16,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '../input/chest_xray/chest_xray/val',
        target_size=(128, 128),
        batch_size=16,
        class_mode='binary')
train_generator.class_indices
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Dropout
classifier =Sequential()

# Convolution
classifier.add(Conv2D(128, (3, 3), input_shape = (128, 128,3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding  second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding third  convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.15))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.20))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
classifier.fit_generator(
        train_generator,
        steps_per_epoch=300,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=16)
test_accuracy = classifier.evaluate_generator(test_gen,steps=624)
print ( "Test Loss ")
print(test_accuracy[0])
print("Test Accuracy")
print(test_accuracy[1])
# Classification Report
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix


predictions = classifier.predict_generator(test_gen)
pred_labels = (predictions > 0.5).astype(np.int)
true_labels = test_gen.classes
class_labels = list(test_gen.class_indices.keys()) 
print('Confusion Matrix')
print(confusion_matrix(true_labels, pred_labels))
report = metrics.classification_report(true_labels, pred_labels, target_names=class_labels)
print(report) 
# Get the confusion matrix
from mlxtend.plotting import plot_confusion_matrix
cm  = confusion_matrix(true_labels, pred_labels)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Reds)
plt.xticks(range(2), ['Normal', 'Pneumonia'])
plt.yticks(range(2), ['Normal', 'Pneumonia'])
plt.show()
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Recall of the model is ",(recall))
print("Precision of the model ",(precision))