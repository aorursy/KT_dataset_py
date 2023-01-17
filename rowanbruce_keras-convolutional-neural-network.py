import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt



train = pd.read_csv("../input/train1/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train.head()
a = train.label

train_labels = a.to_frame()



#important to have pixels np arrays to use "reshape" 

train_pixels = train.drop('label', 1).values

test_pixels = test.values
#reshape to input into cnn as [w][h[d]

train_pixels = train_pixels.reshape(train_pixels.shape[0], 28, 28,1).astype('float')

test_pixels = test_pixels.reshape(test_pixels.shape[0], 28, 28,1).astype('float32')
train_pixels.shape
train_pixels = train_pixels/255
train_labels.shape
train_labels = to_categorical(train_labels)

num_classes = train_labels.shape[1]



train_labels
train_labels.shape
model = Sequential()



#add convolutional layer

#32 kernals per conv layer, size of the kernals is 5x5

#input_shape [width][height][depth]

model.add(Conv2D(filters = 32,kernel_size = (5, 5), input_shape=(28, 28, 1), activation='relu'))



#add pooling layer

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



#adding 2nd Conv layer!

model.add(Conv2D(32, (3, 3),activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



#3rd conv

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))



#add dropout layer, excludes 20% of neurons to avoid overfiting

model.add(Dropout(0.3))



#converts 2d matrix to vector... allows the output to be processed by standard fully connected layers.

model.add(Flatten())



#adds a fully connected layer with 256 neurons

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))



model.add(Dense(128, activation='relu'))



model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_labels.shape

model.fit(train_pixels, train_labels, epochs=10, batch_size=200, verbose=2)

predictions = model.predict_classes(test_pixels)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.head()

submissions.to_csv("result.csv", index=False, header=True)

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





datagen.fit(train_pixels)
model.fit_generator(datagen.flow(train_pixels,train_labels, batch_size=200),epochs =10, verbose=2)
predictions = model.predict_classes(test_pixels)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("result.csv", index=False, header=True)
