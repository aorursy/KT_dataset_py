import numpy as np 
import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

print(os.listdir("../input"))
train= '../input/chest-xray-pneumonia/chest_xray/train/'
test = '../input/chest-xray-pneumonia/chest_xray/test/'
val = '../input/chest-xray-pneumonia/chest_xray/val/'
train_normal = train+'NORMAL/'
train_pneumonia = train+'PNEUMONIA/'
rand1 = np.random.randint(0,len(os.listdir(train_normal)))
img1 = train_normal+os.listdir(train_normal)[rand1]
print('Normal picture is : ',img1)

rand2 = np.random.randint(0,len(os.listdir(train_pneumonia)))
img2 = train_pneumonia+os.listdir(train_pneumonia)[rand2]
print('Normal picture is : ',img2)

fig = plt.figure(1, figsize = (15,7))

f1 = fig.add_subplot(1,2,1)
plt.imshow(Image.open(img1),cmap = 'gray')
f1.set_title('Normal')

f2 = fig.add_subplot(1,2,2)
plt.imshow(Image.open(img2),cmap = 'gray')
f2.set_title('Pneumonia')
li = []
for i in range(0,len(os.listdir(train_normal))):
    li.append('Normal')
for i in range(0,len(os.listdir(train_pneumonia))):
    li.append('Pneumonia')
sns.countplot(li)
# Initialising the CNN
classifier = Sequential()

# First CNN layer
classifier.add(Convolution2D(32,(3,3), input_shape = (128,128,3),
                            activation = 'relu'))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Second CNN layer
classifier.add(Convolution2D(32,(3,3),activation = 'relu'))
classifier.add(Dropout(0.1))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Third CNN layer
classifier.add(Convolution2D(32,(3,3),activation = 'relu'))
classifier.add(Dropout(0.2))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Flattening
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
classifier.summary()
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# for training set
training_set = train_datagen.flow_from_directory(train,
                                                 target_size = (128,128),
                                                 batch_size = 16,
                                                 class_mode = 'binary')


# for test set
test_set = test_datagen.flow_from_directory(test,
                                            target_size = (128,128),
                                            batch_size = 16,
                                            class_mode = 'binary')
# for validation set
validation_set = test_datagen.flow_from_directory(val,
                                            target_size = (128,128),
                                            batch_size = 16,
                                            class_mode = 'binary')
history = classifier.fit_generator(training_set,
                         steps_per_epoch = (5217//16),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (625//16))
classifier.evaluate_generator(test_set)
fig1,ax1 = plt.subplots(1,2,figsize = (20,5))
    
for i,j in enumerate(['accuracy','loss']):
    ax1[i].plot(history.history[j],'o-',color = 'green')
    ax1[i].plot(history.history['val_'+j],'o-',color = '#D66520')
    ax1[i].set_title('Model '+str(j))
    ax1[i].set_xlabel('epochs')
    ax1[i].set_ylabel(j)
    ax1[i].legend(['train','val'])
scores = classifier.evaluate_generator(test_set)
print('Accuracy of the Model is '+str(scores[1] *100))
print('Loss of the Model is '+str(scores[0]))
