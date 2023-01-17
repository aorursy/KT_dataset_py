import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras



# regularization

from keras import regularizers



# optimizer

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import LearningRateScheduler



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
print("train shape: ", train.shape)

print("test shape: ", test.shape)
train.head()
test.head()
X = train.drop(['label'], axis = 1).values/255

Y = train['label'].values



X_valid = test.values/255



# reshape

X = X.reshape(X.shape[0],28,28,1)

X_valid = X_valid.reshape(X_valid.shape[0],28,28,1)

#Y = tf.keras.utils.to_categorical(Y)
print("train shape: ", X.shape)

print("test shape: ", X_valid.shape)

print("target shape: ", Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_dev, Y_train,Y_dev = train_test_split(X,Y,test_size = 0.2)
plt.imshow(X_train[1][:,:,0])

plt.title(str(Y_train[1]))
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X[i][:,:,0], cmap=plt.cm.binary)

    plt.xlabel(train.label[i])

plt.show()
f = 2



model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(f*16,kernel_size = (3,3), padding = 'same',activation='relu',

                           kernel_initializer='he_uniform',

                           input_shape = (28,28,1)),

    tf.keras.layers.Conv2D(f*16, (3,3), activation = "relu", padding = 'same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    

    tf.keras.layers.Conv2D(f*32, kernel_size = (3,3), padding = 'same',activation='relu'),

    tf.keras.layers.Conv2D(f*32, (3,3), activation = "relu", padding = 'same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Conv2D(f*64, kernel_size = (3,3), padding = 'same',activation='relu'),

    tf.keras.layers.Conv2D(f*64, (3,3), activation = "relu", padding = 'same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.125),

    

    keras.layers.Flatten(),

    keras.layers.Dense(512, activation = 'relu'),

    keras.layers.Dense(128, activation = 'relu'),

    keras.layers.Dense(10, activation='softmax')

    

])



model.summary()
from keras.utils import plot_model

#tf.keras.utils.plot_model(

#    model,

#    to_file='model.png',

#    show_shapes=True,

#    show_layer_names=True,

#    rankdir='TB',

#    expand_nested=False,

#    dpi=96

#)
model.compile(optimizer=Adam(learning_rate=0.0003),

             loss = 'sparse_categorical_crossentropy',

             metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import LearningRateScheduler



train_datagen = ImageDataGenerator(rotation_range=10,

                                   width_shift_range=0.15,

                                   height_shift_range=0.15,

                                   shear_range=0.05,

                                   zoom_range=0.15,

                                   horizontal_flip=False)



valid_datagen = ImageDataGenerator(horizontal_flip=False,

                                    #rotation_range=10,

                                    #width_shift_range=0.25,height_shift_range=0.2,

                                    #shear_range=0.1,zoom_range=0.25

                                    )



# add early stopping

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)



# learning rate decay

def lr_decay(epoch, initial_learningrate = 0.0003):#lrv

    return initial_learningrate * 0.99 ** epoch



# Set a learning rate annealer

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 

                                            patience=300, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# fit model with generated data

batchsize = 512*2

epoch = 45



history = model.fit_generator(train_datagen.flow(X, Y, batch_size = batchsize),

                   steps_per_epoch = 100, 

                    epochs = epoch,

                   callbacks=[#learning_rate_reduction, 

                              LearningRateScheduler(lr_decay),

                              callback],

                   validation_data=valid_datagen.flow(X_dev, Y_dev),

                    validation_steps=50,

                   )
# add early stopping

#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)



# train model

#history = model.fit(X_train, Y_train, epochs=30,

#          callbacks=[callback],

#          validation_data=[X_dev,Y_dev]

#         )



# fits the model on batches with real-time data augmentation:

#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),

#                    steps_per_epoch=len(X) // 32, epochs=45,

                    #callbacks = [callback],

#                    validation_data = (X_dev, Y_dev)

#                   )



# evaluate model performance

test_loss, test_acc = model.evaluate(X_dev, Y_dev,verbose=2)



print('\nTest accuracy: ', test_acc)

print('\nTest loss: ', test_loss)
yhat = model.predict_classes(X_valid)

submission['Label']=pd.Series(yhat)

submission.to_csv('submission.csv',index=False)
fig = plt.figure(figsize = (10,5))

plt.subplot(1,2,1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train','Validation'])



plt.subplot(1,2,2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train','Validation'])