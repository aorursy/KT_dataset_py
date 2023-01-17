import pandas as pd

import numpy as np

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot  as plt

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization

from tensorflow.keras.models import Sequential, load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

tf.__version__

train = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv')

train.shape, test.shape
X_train = x_train = train.drop(['label'],1)

Y_train = train['label']

x_test = test
X_train = np.asarray(X_train)

x_test = np.asarray(x_test)

g = sns.countplot(Y_train)

x_train.shape
X_train = X_train.astype('float32')

x_test = x_test.astype('float32')



X_train = X_train/255

x_test - x_test/255
Y_train.shape
# one hot encoding y data

Y_train= tf.keras.utils.to_categorical(Y_train, 10)

Y_train.shape
X_train = X_train.reshape(-1, 28, 28, 1)

x_test = x_test.reshape(-1,28 ,28, 1)

X_train.shape
x_train, val_x, y_train, val_y = train_test_split(X_train, Y_train, test_size=0.20)
val_x.shape
es = EarlyStopping(monitor='loss', patience=12)

filepath="/kaggle/working/bestmodel.h5"

md = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# defininig ImageDataGeneratore to increase data

datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
# Important Variables

epochs = 30

num_classes = 10

batch_size = 30

input_shape = (28, 28, 1)

adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model = Sequential()



# Filter 1

model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation= tf.nn.relu)) 

model.add(Conv2D(32, (3, 3), padding='same', activation= tf.nn.relu))    

#model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))





# Filter 2

model.add(Conv2D(64, (3, 3), padding='same', activation= tf.nn.relu))                          

#model.add(Conv2D(64, (3, 3), padding='same', activation= tf.nn.relu))    

#model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))







'''# Filter 3 

model.add(Conv2D(64, (3, 3), padding='same', activation= tf.nn.relu))                         

model.add(Conv2D(64, (3, 3), padding='same', activation= tf.nn.relu))                        

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(BatchNormalization())'''

















# Dense Filter 5

model.add(Flatten())

model.add(Dense(1024, activation=tf.nn.relu))                                                

model.add(Dropout(0.25))

#model.add(BatchNormalization())





# Dense Filter 1



'''model.add(Dense(512, activation=tf.nn.relu))                                                

model.add(Dropout(0.25))'''

#model.add(BatchNormalization())



# Dense Filter 2       

model.add(Dense(512, activation=tf.nn.relu))                                                                 

model.add(Dropout(0.25))

#model.add(BatchNormalization())



# Dense Filter 3

model.add(Dense(256, activation=tf.nn.relu))                                                   

model.add(Dropout(0.5))

#model.add(BatchNormalization())



# Dense Filter 4

model.add(Dense(10, activation= tf.nn.softmax))                                                  



# Model Compile

model.compile(optimizer= adam, loss= tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])



# Model Summery

model.summary()

History = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

          #steps_per_epoch=840,

          epochs = epochs,

          validation_data = (val_x, val_y),

          callbacks = [es,md],

          shuffle= True

        )

       
fig, ax = plt.subplots(2,1)

ax[0].plot(History.history['loss'], color='b', label="Training loss")

ax[0].plot(History.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=False)



ax[1].plot(History.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(History.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=False)
model1 = load_model("/kaggle/working/bestmodel.h5")
model1.summary()
pred = model1.predict(x_test)

pred_class = model1.predict_classes(x_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred_class)+1)),

                         "Label": pred_class})

submissions.to_csv("submissions.csv", index=False, header=True)

submissions