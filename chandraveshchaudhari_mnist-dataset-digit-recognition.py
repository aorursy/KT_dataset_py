import os

os.listdir()
os.chdir('/kaggle/input')
os.listdir()
import pandas as pd
traindatafile = pd.read_csv('train.csv')

testdatafile = pd.read_csv('test.csv')
traindatafile.head()

y = traindatafile.pop('label')
import numpy as np # linear algebra

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
x_train, x_val, y_train, y_val = train_test_split(traindatafile, y, test_size=0.1, random_state=42)
traindatafile.shape
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

x_train = x_train.values.reshape(-1,28,28,1)

x_val = x_val.values.reshape(-1,28,28,1)

x_test= testdatafile.values.reshape(-1,28,28,1)
print(x_train.shape)

print(x_val.shape)
x_train[0]
x_train = x_train.astype("float32")/255.

x_val = x_val.astype("float32")/255.

x_test = x_test.astype("float32")/255.
x_train[0]
print(x_train.shape)

x_val.shape
x_test.shape
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.callbacks import TensorBoard
y_train = to_categorical(y_train) 

y_val = to_categorical(y_val) 
print(y_train.shape)

y_val.shape
import matplotlib.pyplot as plt

%matplotlib inline 



g = plt.imshow(x_train[0][:,:,0])
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.summary()
datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

                    samplewise_center=False,  # set each sample mean to 0

                                 featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

                               zca_whitening=False,  # apply ZCA whitening

        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)

                              zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

                               height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

                               vertical_flip=False)  # randomly flip images)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"]) #1e-4, means the 1 is four digits the other way, so 1e-4 = 0.0001.
learning_rate_min = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

#tensor= TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq=1000)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),

                           steps_per_epoch=500,

                           epochs=20, #Increase this when not on Kaggle kernel

                           verbose=2,  #1 for ETA, 0 for silent

                           validation_data=(x_val[:400,:], y_val[:400,:]), #For speed

                           callbacks=[learning_rate_min]) 
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
plt.plot(hist.history['loss'], color='r',label='Training loss')

plt.plot(hist.history['acc'], color='b',label='Training accuracy')

plt.title('Training loss and Training accuracy')

plt.show()



plt.plot(hist.history['val_loss'], color='r', label='Validation loss')

plt.plot(hist.history['val_acc'], color='b', label='Validation accuracy')

plt.title('validation loss and validation accuracy')

plt.show()
y_predicted = model.predict(x_val)

y_predictedint = np.argmax(y_predicted, axis=1)

y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_predictedint)

print(cm)
y_hat = model.predict(x_test ,batch_size=64)
print(y_hat)
y_pred = np.argmax(y_hat,axis=1)
y_pred
print(y_pred)
y_pred.shape