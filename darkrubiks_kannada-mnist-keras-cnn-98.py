import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv("../input/Kannada-MNIST/train.csv") 

test = pd.read_csv("../input/Kannada-MNIST/test.csv")
y = train["label"]

train.drop("label",axis=1,inplace=True) 

test.drop("id",axis=1,inplace=True)
x_train = train.values

test = test.values



x_train = x_train/255

test = test/255
x_train = x_train.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)
y_train = pd.get_dummies(y)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=42,stratify=y_train)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Dropout,MaxPool2D,BatchNormalization

from keras.preprocessing. image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import RMSprop
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
checkpoint=ModelCheckpoint('bestweights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode="max")
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])
train_gen = ImageDataGenerator(rotation_range=10, 

                               width_shift_range=0.1, 

                               height_shift_range=0.1, 

                               zoom_range=0.1) 
history= model.fit_generator(train_gen.flow(x_train,y_train,batch_size=86),

                             steps_per_epoch=x_train.shape[0]//86,

                             validation_data=(x_test,y_test),

                             validation_steps=x_test.shape[0]//86,

                             epochs=30,callbacks=[checkpoint,learning_rate_reduction ])
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.load_weights("bestweights.hdf5")
results = model.predict(test)

results = np.argmax(results,axis = 1)

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = results

submission.to_csv('submission.csv', index=False)

submission.head()