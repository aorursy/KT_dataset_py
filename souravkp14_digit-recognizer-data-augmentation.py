import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline
#load data
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print('The shape of training dataset : ', train_df.shape)
print('The shape of testing dataset : ', test_df.shape)
Y_train = train_df["label"]
X_train = train_df.drop(labels = ["label"],axis = 1) 

count = sns.countplot(Y_train)

Y_train.value_counts()
#The data right now is in an int8 format, so before you feed it into the network you need to convert its type to float32
#you also have to rescale the pixel values in range 0 - 1 inclusive

X_train = X_train.astype('float32')
test = test_df.astype('float32')
X_train = X_train / 255.
test = test / 255.

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

X_train.shape, test.shape
# Change the labels from categorical to one-hot encoding
Y_train_one_hot = to_categorical(Y_train)

# Display the change for category label using one-hot encoding
print('Original label:', Y_train[0])
print('After conversion to one-hot:', Y_train_one_hot[0])
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(X_train, Y_train_one_hot, test_size=0.2, random_state=21)
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
batch_size = 128
epochs = 20
num_classes = 10
#Neural network architecture

model = Sequential()


model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
# Define the optimizer
from keras.optimizers import RMSprop

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer ,metrics=['accuracy'])
model.summary()
#Data augmentation

from keras.preprocessing.image import ImageDataGenerator

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
            horizontal_flip=False,  # randomly flip images, can misclassify symetrical numbers
            vertical_flip=False)  # randomly flip images, can misclassify symetrical numbers
datagen.fit(train_X, augment=True)



train_model = model.fit_generator(datagen.flow(train_X, train_label,batch_size=batch_size), epochs=epochs, verbose=1,validation_data=(valid_X, valid_label))
#plotting the accuracy and loss plots between training and validation data

accuracy = train_model.history['accuracy']
val_accuracy = train_model.history['val_accuracy']
loss = train_model.history['loss']
val_loss = train_model.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
test_eval = model.evaluate(valid_X, valid_label, verbose=1)
print('Validation Test loss:', test_eval[0])
print('Validation Test accuracy:', test_eval[1])
# predict results
predict_classes = model.predict(test)

# # select the indix with the maximum probability
predict_classes = np.argmax(np.round(predict_classes),axis=1)

results = pd.Series(predict_classes,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)

