import pandas as pd

import numpy as np

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/train.csv')
train.shape
y_train=train['label']

y_train=np.array(y_train)





train=train.drop('label',axis=1)

train=np.array(train)

train=train.reshape((42000,28,28,1))

train=train.astype('float32')/255







train.shape
y_test=test['label']
test=test.drop('label',axis=1)
test=np.array(test)

test=test.reshape((42000,28,28,1))

test=test.astype('float32')/255
from keras import callbacks
from keras import layers

from keras.models import Sequential

from keras import optimizers

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
y_train=to_categorical(y_train)


train, val, y_train, y_val = train_test_split(train, y_train, test_size = 0.1, random_state=2)

model = Sequential()



model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(layers.Dropout(0.5))





model.add(layers.Flatten())

model.add(layers.Dense(256, activation = "relu"))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation = "softmax"))

optimizer=optimizers.RMSprop(lr=0.001,decay=0.0,epsilon=1e-08,rho=0.9)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 30# Turn epochs to 30 to get 0.9967 accuracy

batch_size = 86
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(

       

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        shear_range=0.1

        )
history = model.fit_generator(datagen.flow(train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (val,y_val),

                              verbose = 2, steps_per_epoch=train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
model.save('k.h5')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
test.shape

test=test.drop('label',axis=1)

test=np.array(test)

test=test.reshape((42000,28,28,1))
results=model.predict(test)

labels=[]

for i in range(len(results)):

    a=list(results[i])

    labels.append(a.index(max(a)))

labels=pd.DataFrame(labels)
import os

os.mkdir('output')
labels.to_csv(r'output/test_labels.csv')
test