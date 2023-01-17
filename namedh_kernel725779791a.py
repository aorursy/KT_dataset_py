import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers.core import Layer
from keras.initializers import RandomNormal
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop, SGD
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical,plot_model
Y = np.genfromtxt('../input/test.csv',delimiter=',')
Y = Y[1::]
TestData = []
for X in Y:
    inter = np.resize(X,(28,28,1))
    TestData.append(inter)
TestData = np.array(TestData)    
X = np.genfromtxt('../input/train.csv',delimiter=',')
y = X[1::]
print(y.shape)
labels = []
data = []
for x in y:
    labels.append(x[0])
    inter = np.resize(x[1::],(28,28,1))
    data.append(inter)
TrainingData = np.array(data)
ValidationData = TrainingData[36000::]
TrainingData = TrainingData[0:36000]
TrainingLabels = np.array(labels)
TrainingLabels = to_categorical(TrainingLabels)
ValidationLabels = TrainingLabels[36000::]
TrainingLabels = TrainingLabels[0:36000]

batchSize = 256
epoch = 5
optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
numClasses = 10
Initializer1 = RandomNormal(mean=0.0, stddev=0.01, seed=None)
model = Sequential([
    Conv2D(96,(7,7),strides = (2,2), activation = "relu", use_bias = True, bias_initializer='zeros',padding="same",kernel_initializer=Initializer1,input_shape=(28,28,1,)),
    MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    Dropout(0.25),
    Conv2D(256,(5,5),strides =(1,1),kernel_initializer=Initializer1, activation = "relu",padding="same", use_bias = True, bias_initializer='ones'),
    Conv2D(256,(5,5),strides =(1,1),kernel_initializer=Initializer1, activation = "relu",padding="same", use_bias = True, bias_initializer='ones'),
    MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
    Dropout(0.25),
    Conv2D(384,(3,3),strides =(1,1),kernel_initializer=Initializer1,padding="same", use_bias = True, bias_initializer='zeros'),
    Conv2D(256,(3,3),strides =(1,1),kernel_initializer=Initializer1,padding="same", use_bias = True, bias_initializer='ones'),
    MaxPooling2D(pool_size=(3, 3), strides=(2,2),padding="same"),
    BatchNormalization(),
    Flatten(),
    Dense(256, activation='relu',kernel_initializer=Initializer1, use_bias = True, bias_initializer='ones'),
    Dropout(0.25),
    Dense(10,kernel_initializer=Initializer1, use_bias = True, bias_initializer='ones', activation='softmax'),
])
model.summary()
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(TrainingData)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])
history = model.fit_generator(datagen.flow(TrainingData,TrainingLabels, batch_size=batchSize),
                              epochs = 30,validation_data=(ValidationData, ValidationLabels),
                              verbose = 1)
classes = model.predict(TestData, batch_size=256)
results = []
for x in classes:
    results.append(x.tolist().index(max(x)))
numbers = range(1,28001)
import pandas as pd
my_submission = pd.DataFrame({'ImageId': numbers, 'Label': results})
my_submission.to_csv('submission.csv', index=False)

classes = model.predict(ValidationData, batch_size=256)
results = []
for x in classes:
    results.append(x.tolist().index(max(x)))
results2 = []
for x in ValidationLabels:
    results2.append(x.tolist().index(max(x)))
print('confusion matrix of Validation Data')
print(confusion_matrix(results2,results))
