import numpy as np

from sklearn.datasets import load_files



train = '../input/fruits/fruits-360/Training'

test = '../input/fruits/fruits-360/Test'



def split(inp):

    i = load_files(inp)

    j = np.array(i['filenames'])

    k = np.array(i['target'])

   

    return j,k

    

x_train, y_train = split(train)

print('Training = ' , x_train.shape)

x_test, y_test = split(test)

print('Testing = ', x_test.shape)







classes = len(np.unique(y_train))

classes




from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train,classes)

y_test = np_utils.to_categorical(y_test,classes)
from keras.preprocessing.image import array_to_img, img_to_array, load_img



def array(files):

    images_as_array=[]

    for file in files:

        # Convert to Numpy Array

        images_as_array.append(img_to_array(load_img(file)))

    return images_as_array



x_train = np.array(array(x_train))

print('Training set shape : ',x_train.shape)



x_test = np.array(array(x_test))

print('Test set shape : ',x_test.shape)

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D

from keras.layers import Activation, Dense, Flatten, Dropout

from keras.layers.normalization import BatchNormalization





model = Sequential()



model.add(Conv2D(16, (3, 3), input_shape=(100,100,3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(16, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(32,(3,3 )))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Flatten())



# Fully connected layer

model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.8))

model.add(Dense(103))



model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



#np.resize(img, (-1, <image shape>)

#x_trai = x_train.reshape(,100, 100, 3)



output = model.fit(x_train,y_train,

        batch_size = 32,

        epochs=10)