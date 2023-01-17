import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for plotting





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#loading the dataset.......(Train)

train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
#loading the dataset.......(Test)

test= pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
x_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

x_test = test.values.astype('float32')



print(x_train)

print(y_train.shape)

print(x_test.shape)
#scale The Data

x_train = x_train/255.0

x_test = x_test/255.0
train.head()
test.head()
print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
#reform the data

X_train = x_train.reshape(x_train.shape[0], 28, 28,1)

X_test = x_test.reshape(x_test.shape[0], 28, 28,1)

print(X_train.shape)

print(X_test.shape)
#after data ready



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

batch_size = 64

num_classes = 10

epochs = 15

input_shape = (28, 28, 1)
# convert class vectors to binary class matrices One Hot Encoding

#del X_train, X_val, Y_train, Y_val

y_train = keras.utils.to_categorical(y_train, num_classes)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)

print(X_train.shape)

print(X_val.shape)

print(Y_train.shape)

print(Y_val.shape)
model = Sequential()

model.add(Conv2D(32,(3,3), activation='relu',padding='same',input_shape=input_shape))

model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(.2))



model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(.2))



model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(.2))

model.add(Dense(num_classes,activation='softmax'))





model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.RMSprop(),

              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



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

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
model
datagen.fit(X_train)

h = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
print(h.history.keys())

#  "Accuracy"

plt.plot(h.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(h.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("asd.csv", index=False, header=True)
print(check_output(["ls", "./"]).decode("utf8"))
result = pd.read_csv("./asd.csv")

print(result.shape)

result.head()
print(result)
model.save("model1.save")
print(check_output(["ls", "./"]).decode("utf8"))
from subprocess import run

!ls -lah