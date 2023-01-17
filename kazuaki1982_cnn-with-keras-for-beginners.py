# import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,BatchNormalization,Dropout,Flatten,Dense,ZeroPadding2D

from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping,ReduceLROnPlateau



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# load train and test data

df_train = pd.read_csv('/kaggle/input/fashion-mnist_train.csv')

df_test = pd.read_csv('/kaggle/input/fashion-mnist_test.csv')



X_train = df_train.iloc[:,1:]

y_train = df_train.iloc[:,0]

X_test = df_test.iloc[:,1:]

y_test = df_test.iloc[:,0]



# convert to numpy

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

y_train = y_train.values



print('X_train shape:',X_train.shape)

print('X_test shape:',X_test.shape)
plt.figure(figsize=(14,14))

for i in range(36):

    plt.subplot(6,6,i+1)

    plt.title('X_train#{},label{}'.format(i,y_train[i]))

    plt.tick_params(

        axis='x',         

        which='both',     

        bottom=False,      

        top=False,        

        labelbottom=False) 

    plt.imshow(X_train[i,:,:,0],cmap='binary')

plt.show()
df_train.iloc[:,0].value_counts()
# convert label to one-hot encoding

y_train = to_categorical(y_train)
model = Sequential()

model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(64,(3,3),padding='same',activation='relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(256,(3,3),padding='same',activation='relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(128,(3,3),padding='same',activation='relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.3))



# model.add(Conv2D(256,(3,3),padding='same',activation='relu'))

# model.add(MaxPool2D(2,2))

# model.add(BatchNormalization())

# model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',

             optimizer=RMSprop(),

             metrics=['acc'])
# callbacks

es=EarlyStopping(patience=10,verbose=1)

rop=ReduceLROnPlateau(patience=4,verbose=1)
from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1./255,

#                               rotation_range=10,

#                               width_shift_range=0.2,

#                               height_shift_range=0.2,

#                               shear_range=0.1,

#                               zoom_range=0.1,

                              validation_split=0.25)

train_generator=train_gen.flow(X_train,y=y_train,subset='training')

valid_generator=train_gen.flow(X_train,y=y_train,subset='validation')
history=model.fit_generator(train_generator,

                            epochs=100,

                            steps_per_epoch=np.floor(X_train.shape[0]*0.75/32),

                            validation_data=valid_generator,

                            validation_steps=np.floor(X_train.shape[0]*0.25/32),

                           callbacks=[es,rop])


accuracy = history.history['acc']

val_accuracy = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

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
#get the predictions for the test data

X_test =X_test/255.

predicted_classes = model.predict_classes(X_test)



#get the indices to be plotted

correct = np.nonzero(predicted_classes==y_test)[0]

incorrect = np.nonzero(predicted_classes!=y_test)[0]
from sklearn.metrics import accuracy_score

accuracy_score(predicted_classes,y_test)