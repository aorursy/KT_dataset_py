#Importing necessary libraries

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import keras
train = pd.read_csv('../input/train.csv')
train.head()
test = pd.read_csv('../input/test.csv')
train.shape, test.shape
#Randomly shuffling the training set (incase consecutive digits are same)

indices = np.arange(len(train))

np.random.shuffle(indices)

trainShuffled = train.loc[indices,:]
#A visual plot of the images

for i in range(25):

  plt.subplot(5,5,i+1)

  plt.imshow(np.array(trainShuffled.iloc[i,1:]).reshape(28,28), cmap = 'gray')

  plt.axis('off')


#Creating Keras Model, consiting of Conv2d + Pooling blocks followed by one Dense layer with 10 hidden units

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3,3), input_shape = (28,28,1,), activation = 'relu', name = 'Conv2D1'))

model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', name = 'Conv2D2'))

model.add(keras.layers.MaxPooling2D(2,2))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(128, (2,2),activation = 'relu', name = 'Conv2D3'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(10, activation = 'softmax'))
model.summary()
from keras.preprocessing.image import ImageDataGenerator
generator = ImageDataGenerator(rescale = 1./255, rotation_range = 15, width_shift_range = 0.1, height_shift_range = 0.1,

                              shear_range = 0.1, zoom_range = 0.1)
#Validation set and train set split

train_X = trainShuffled.iloc[:35000, 1:]

train_y = trainShuffled.iloc[:35000, 0]

test_X = trainShuffled.iloc[35000:, 1:]

test_y = trainShuffled.iloc[35000:, 0]

train_X.shape, train_y.shape, test_X.shape, test_y.shape
train_X = np.array(train_X).reshape(-1,28,28,1) #Networks expects a 4D input

                                        #(samples, height ,width, channel), since not RGB channel = 1

test_X = np.array(test_X).reshape(-1,28,28,1)
train_X.shape, test_X.shape
model.compile(optimizer = keras.optimizers.Adam(.001), loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy']) #Our y_train is numerical, so instead of categorical_crossentropy

                                      #we use sparse_categorical_cross_entropy, and a learning rate 0.001
myDataGen = generator.flow(train_X,train_y, batch_size = 100) #We get data augmented images
test_X = test_X/255. #Also normalize test data, since we did it with train data
hist  = model.fit_generator(myDataGen, steps_per_epoch = 350, epochs = 40, validation_data = (test_X, test_y))
hist.history.keys()
val_loss = np.array(hist.history['val_loss'])

val_acc = np.array(hist.history['val_acc'])

train_loss = np.array(hist.history['loss'])

train_acc = np.array(hist.history['acc'])

epochs = len(train_acc)
accuracies = pd.DataFrame(train_acc, columns = ['train_acc'])

accuracies['val_acc'] = pd.DataFrame(val_acc)

losses = pd.DataFrame(train_loss, columns = ['train_loss'])

losses['val_loss'] = pd.DataFrame(val_loss)

losses['epochs'] = pd.DataFrame(list(range(epochs)))

accuracies['epochs'] = pd.DataFrame(list(range(epochs)))
accuracies.shape


plt.scatter(x = 'epochs', y = 'train_acc', data = accuracies, marker = 'x', label = 'train_acc')

plt.scatter(x = 'epochs', y = 'val_acc', s = 5,data = accuracies, label = 'val_acc', color = 'r')

plt.plot(accuracies.iloc[:,0:2])

plt.legend()

plt.show()



plt.scatter(x = 'epochs', y = 'train_loss', data = losses, marker = 'x', label = 'train_loss')

plt.scatter(x = 'epochs', y = 'val_loss', data = losses, s = 5,label = 'val_loss', color = 'r')

plt.plot(losses.iloc[:,0:2])

plt.legend()

plt.show()

test = np.array(test)/255. #Need also to normalize data that we need to predict

test = test.reshape(-1,28,28,1)
#Combine train and validation set for final model training, from scratch

totalData_X = np.concatenate((train_X, test_X))

totalData_y = np.concatenate((train_y, test_y))

myDataGen = generator.flow(totalData_X,totalData_y, batch_size = 100)
#Redefining model

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3,3), input_shape = (28,28,1,), activation = 'relu', name = 'Conv2D1'))

model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', name = 'Conv2D2'))

model.add(keras.layers.MaxPooling2D(2,2))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(128, (2,2),activation = 'relu', name = 'Conv2D3'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(10, activation = 'softmax'))
model.compile(optimizer = keras.optimizers.Adam(.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
hist  = model.fit_generator(myDataGen, steps_per_epoch = 450, epochs = 40)
preds = model.predict(test)
predict = pd.DataFrame([preds[x].argmax() for x in range(len(preds))], columns = ['Label'])

predict.head(2)

sample = pd.read_csv('../input/sample_submission.csv')
predict['ImageId'] = sample['ImageId']

predict = predict[['ImageId','Label']]
predict.head()
predict.to_csv('predictionsMy.csv', index = False)