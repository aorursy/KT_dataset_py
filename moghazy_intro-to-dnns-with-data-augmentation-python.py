from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import pandas as pd

from keras.datasets import mnist

import numpy as np

np.random.seed(0)



x_train = pd.read_csv('../input/train.csv')

label = x_train['label']

x_train.drop(['label'], inplace = True, axis = 1 )



x_test = pd.read_csv('../input/test.csv')

x_train = x_train.values

y_train = label.values

x_test = x_test.values

x_train ,  y_train = shuffle(x_train, label , random_state=0)
print(x_train.shape)
print("the number of training examples = %i" % x_train.shape[0])

print("the number of classes = %i" % len(np.unique(y_train)))

print("Flattened Image dimentions = %d x %d  " % (x_train.shape[1], 1)  )



#This line will allow us to know the number of occurrences of each specific class in the data

print("The number of occuranc of each class in the dataset = %s " % label.value_counts(), "\n" )





X_train = x_train.reshape(-1, 28, 28).astype('float32')

images_and_labels = list(zip(X_train,  y_train))

for index, (image, label) in enumerate(images_and_labels[:12]):

    plt.subplot(5, 4, index + 1)

    plt.axis('off')

    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('label: %i' % label)
type(x_train)
from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.layers import Dropout, Conv2D

from keras import regularizers



from keras.utils import np_utils



#reshape the inputs

# I will change the size of the training and testing sets to be able to use ImageDataGenerator wich accepts inputs in the following shape

x_train = x_train.reshape(-1,28,28,1)

x_test = x_test.reshape(-1,28,28,1)



print(x_train.shape )

print(x_train.shape )





#Makine the outputs 1-hot vector of 10 elements

y_train = np_utils.to_categorical(y_train)



model = Sequential()

# The first layer doesn't have significant importance in the code.

# THe conv layer is used only to get the 3d images from the fit generator in the 2d format and flatten it using flatten layer

# THe layer will not affect the layer since i am only using feature Pooling _ 1*1 convolution with only 1 feature map

model.add(Conv2D(1, kernel_size=1, padding="same",input_shape=(28, 28, 1), activation = 'relu'))

model.add(Flatten())



# model.add(Dense(units=800, activation='relu', input_dim= 784 ,  kernel_regularizer=regularizers.l2(0.001) ) )



model.add(Dense(units=100, activation='relu'  ))

model.add(Dropout(0.1))

model.add(Dense(units=100, activation='relu'  ))

model.add(Dropout(0.1))

model.add(Dense(units=100, activation='relu'  ))

model.add(Dropout(0.1))



#and now the output layer which will have 10 units to

#output a 1-hot vector to detect one of the 10 classes

model.add(Dense(units=10, activation='softmax'))
from keras import optimizers



# optimizer = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)

model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator

x_train2 = np.array(x_train, copy=True) 

y_train2 = np.array(y_train, copy=True) 



datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=10,

    fill_mode='nearest',

    validation_split = 0.2

    )





# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)



datagen.fit(x_train)



print(type(x_train))



earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')



validation_generator = datagen.flow(x_train2, y_train2, batch_size=60, subset='validation')

train_generator = datagen.flow(x_train2, y_train2, batch_size=60, subset='training')





# # fits the model on batches with real-time data augmentation:

history = model.fit_generator(generator=train_generator,

                    validation_data=validation_generator,

                    use_multiprocessing=True,

                    steps_per_epoch = len(train_generator) / 60,

                    validation_steps = len(validation_generator) / 60,

                    epochs = 300,

                    workers=-1, callbacks = [earlystopping])
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([-1,1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([-1,1.0])

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
res = model.predict(x_test)

res = np.argmax(res,axis = 1)

res = pd.Series(res, name="Label")

submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)

submission.to_csv("solution.csv",index=False)

submission.head(10)