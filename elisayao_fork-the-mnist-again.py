# check input files

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
train_file = "../input/train.csv"

test_file = "../input/test.csv"

output_file = "submission.csv"
# load training data

                      #path    #skip labels  #specify datatype

raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
# train-test split

#x_train #x_test #y_train #y_test                #predictors    #response(labels)

x_train, x_val, y_train, y_val = train_test_split(raw_data[:,1:],raw_data[:,0],test_size=0.1)
print("Original dataset:")

print("training x: ", "shape ", x_train.shape, " type ", type(x_train))

print("training y: ", "shape ", y_train.shape, " type ", type(y_train))

print("validation x: ", "shape ", x_val.shape, " type ", type(x_val))

print("validation y: ", "shape ", y_val.shape, " type ", type(y_val))
# scale the data

x_train = x_train.astype("float32")/255.0   # convert it to [0,1] scale

x_val = x_val.astype("float32")/255.0

y_train = np_utils.to_categorical(y_train)   # convert integers to dummy variables (one hot encoding) 

y_val = np_utils.to_categorical(y_val)
n_train = x_train.shape[0]    # number of training observations

n_val = x_val.shape[0]    # number of validation observations

                         #obs  #28px #28px #1channel

x_train = x_train.reshape(n_train, 28, 28, 1)

x_val = x_val.reshape(n_val, 28, 28, 1)

n_classes = y_train.shape[1]  # 10 categories
# dimensions of training and testing set after normalization



print("After normalization:")

print("training x: ", "shape ", x_train.shape, " type ", type(x_train))

print("training y: ", "shape ", y_train.shape, " type ", type(y_train))

print("testing x: ", "shape ", x_val.shape, " type ", type(x_val))

print("testing y: ", "shape ", y_val.shape, " type ", type(y_val))
model = Sequential()



# add a convolutional layer

                 #16 filters #filter size    #activation function #input data shape (each ob)

model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))



# add a normalization layer

model.add(BatchNormalization())



# add a convolutional layer

                                                               #don't need to specify the dimension

model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu'))



# add a pooling layer

                    #(2,2) stride value

model.add(MaxPool2D(strides=(2,2)))



# add a normalization layer

model.add(BatchNormalization())



# add a dropout layer

          #fraction of input units to drop

model.add(Dropout(0.25))





# add a convolutional layer

model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))



# add a normalization layer

model.add(BatchNormalization())



# add a convolutional layer

model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))



# add a pooling layer

model.add(MaxPool2D(strides=(2,2)))



# add a normalization layer

model.add(BatchNormalization())



# add a dropout layer

model.add(Dropout(0.25))







# add a flatten layer

model.add(Flatten())



# add a dense layer (fully connected)

model.add(Dense(512,activation='relu'))



# add a dropout layer

model.add(Dropout(0.25))



# add a dense layer (fully connected)

model.add(Dense(1024,activation='relu'))



# add a dropout layer

model.add(Dropout(0.5))



# add a dense layer (output layer, 10 categories)

model.add(Dense(10,activation='softmax'))
# tweek the images

datagen = ImageDataGenerator(zoom_range=0.1,

                             height_shift_range=0.1,

                             width_shift_range=0.1,

                             rotation_range=10)
# compile the model

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=3e-5),metrics=["accuracy"])



hist=model.fit_generator(datagen.flow(x_train,y_train,batch_size=16),

                         steps_per_epoch=1000,

                         epochs=1,

                         verbose=2,

                        validation_data=(x_val[:400,:],y_val[:400,:]))
# reduce learning rate by 10% each epoch

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9**x)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),

                           steps_per_epoch=1000,

                           epochs=9,

                           verbose=2,

                           validation_data=(x_val[:400,:],y_val[:400,:]),

                           callbacks=[annealer])
model.evaluate(x_val, y_val, verbose=0)
# plot the graph of metrics

plt.plot(hist.history['loss'],color='b')

plt.plot(hist.history['val_loss'], color='r')

plt.show()



plt.plot(hist.history['acc'], color='b')

plt.plot(hist.history['val_acc'], color='r')



plt.show()
y_hat = model.predict(x_val)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

print(cm)
# load the test set

mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')

x_test = mnist_testset.astype("float32")/255.0

n_test = x_test.shape[0]

x_test = x_test.reshape(n_test, 28, 28, 1)

y_hat = model.predict(x_test, batch_size=64)
y_pred = np.argmax(y_hat, axis=1)
with open(output_file, 'w') as f:

    f.write('ImageId,Label\n')

    for i in range(0, n_test):

        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))