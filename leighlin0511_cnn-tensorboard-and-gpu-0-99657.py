import keras

keras.__version__
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from sklearn.model_selection import train_test_split



from keras import layers

from keras import models



from keras.callbacks import TensorBoard



from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



from os import makedirs

from os.path import exists, join
train = pd.read_csv('../input/train.csv')

train.head()

test = pd.read_csv('../input/test.csv')
Y_train = train.loc[:]['label']

Y_train = Y_train.values

X_train = train.iloc[:,1:]

del train 

# Normalize the data

X_train = X_train / 255.0

test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
# Set the random seed

random_seed = 3

batch_size=86  



# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val_original = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# Encode labels to one hot vectors (ex : 1 -> [0,1,0,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)

Y_val = to_categorical(Y_val_original, num_classes = 10)
model = models.Sequential()

model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.summary()
epochs=20 

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])



model.fit(X_train, Y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_val, Y_val))

score = model.evaluate(X_val, Y_val, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
datagen = ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1)



datagen.fit(X_train)
log_dir = '../logs'

#Create log_dir if we have premission

if not exists(log_dir):

    makedirs(log_dir)

    

# save class labels to disk to color data points in TensorBoard accordingly

with open(join(log_dir, 'metadata.tsv'), 'w') as f:

    np.savetxt(f, Y_val_original)
tensorboard = TensorBoard(batch_size=batch_size,

                          log_dir = log_dir,

                          embeddings_freq=1,

                          embeddings_layer_names=['features'],

                          embeddings_metadata='metadata.tsv',

                          embeddings_data=X_val)
# Reduce learning rate when a metric has stopped improving

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                            patience=3, 

                            verbose=1, 

                            factor=0.3, 

                            min_lr=0.000001)
model = models.Sequential()

model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', input_shape=(28, 28, 1)))

#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.BatchNormalization()) 

model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.BatchNormalization()) 

model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

model.add(layers.BatchNormalization()) 

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu', name='features'))

model.add(layers.Dense(10, activation='softmax'))
import tensorflow as tf 

tf.test.is_built_with_cuda()

tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
epochs=40 

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])



# Fit the model

model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                epochs = epochs, 

                validation_data = (X_val,Y_val),

                verbose = 1, 

                steps_per_epoch=X_train.shape[0] // batch_size, 

                callbacks=[tensorboard,learning_rate_reduction])

score = model.evaluate(X_val, Y_val, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("../cnn_mnist_submission.csv",index=False)