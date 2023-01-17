#importing libs

import pandas as pd
import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,Activation,Conv2D,MaxPooling2D,Flatten, BatchNormalization, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
#import data

path = "../input/digit-recognizer"
train_dir = os.path.join(path,"train.csv")
test_dir = os.path.join(path,"test.csv")
train = pd.read_csv(train_dir)
test = pd.read_csv(test_dir)
#drop target

x_train = train.drop(columns= "label")
y_train = train["label"]
x_train.head()
y_train.head()
sns.countplot(y_train)
x_train.shape, y_train.shape
#make x_train, x_test 3D for reading images

x_train3d = x_train.values.reshape(x_train.shape[0],28,28)
x_test3d = test.values.reshape(test.shape[0],28,28)
#make x_train, x_test 4D for training

x_train4d = x_train.values.reshape(x_train.shape[0],28,28,1)/255
x_test4d = test.values.reshape(test.shape[0],28,28,1)/255
x_train3d.shape, x_test3d.shape
n = 10
nums = ["0","1","2","3","4","5","6","7","8","9"]
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train3d[i])
    plt.title(nums[y_train[i]])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#preparing data

y_train = to_categorical(y_train,10)
x_train0, x_test0, y_train0, y_test0 = train_test_split(x_train4d,y_train, test_size=0.1,random_state=42)
x_train0.shape,x_test0.shape,y_train0.shape,y_test0.shape
model = Sequential()

input_shape = (28,28,1)
model.add(Conv2D(32, (5, 5), input_shape=input_shape,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(256, (3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer="adam", 
              loss = 'categorical_crossentropy',  
              metrics = ['accuracy'])

model.summary()
#Generate data

gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=23,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.3, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
gen.fit(x_train0)
history = model.fit_generator(gen.flow(x_train0, y_train0, 
                                                 batch_size=512),
                                    epochs = 50,
                                    validation_data = (x_test0,y_test0),
                                    verbose = 2, 
                                    steps_per_epoch=x_train0.shape[0] //512,)
#prediction

y_pred = model.predict(x_test4d)
y_pred = np.argmax(y_pred,axis=1)
#Comparison between predict title and test images

n = 30
nums = ["0","1","2","3","4","5","6","7","8","9"]
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test3d[i])
    plt.title(nums[y_pred[i]])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
my_submission = pd.DataFrame({'ImageId': list(range(1, len(y_pred)+1)), 'Label': y_pred})
my_submission.to_csv('submission.csv', index=False)