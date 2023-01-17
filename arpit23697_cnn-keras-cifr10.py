import sys

import matplotlib.pyplot as plt

from keras.datasets import cifar10

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.optimizers import SGD , Adam



from sklearn.model_selection import train_test_split
# load train and test dataset

def load_dataset():

    # load dataset

    (trainX, trainY), (testX, testY) = cifar10.load_data()

    # one hot encode target values

    trainY = to_categorical(trainY)

    testY = to_categorical(testY)

    return trainX, trainY, testX, testY
train_val_x , train_val_y , test_x , test_y = load_dataset()
train_x , val_x , train_y , val_y = train_test_split(train_val_x , train_val_y , test_size = 0.2 , shuffle = True)
print("Train size", train_x.shape , train_y.shape)

print("Val size", val_x.shape , val_y.shape)

print("Test size", test_x.shape , test_y.shape)
def summarize_diagnostics(history):

    

    plt.figure(figsize = (12 , 6))

    plt.subplot(2 ,1,1)

    

    plt.title('Cross Entropy Loss')

    plt.plot(history.history['loss'], color='blue', label='train')

    plt.plot(history.history['val_loss'], color='orange', label='test')

    plt.legend()

    

    plt.subplot(2,1,2)

    plt.title('Classification Accuracy')

    plt.plot(history.history['accuracy'], color='blue', label='train')

    plt.plot(history.history['val_accuracy'], color='orange', label='test')

    plt.legend()

    plt.show()
# scale pixels

def prep_pixels(train, val, test):

    # convert from integers to floats

    train_norm = train.astype('float32')

    test_norm = test.astype('float32')

    val_norm = val.astype('float32')

    # normalize to range 0-1

    train_norm = train_norm / 255.0

    test_norm = test_norm / 255.0

    val_norm = val_norm / 255.0

    # return normalized images

    return train_norm, val_norm ,test_norm

 
train_x , val_x , test_x = prep_pixels(train_x , val_x , test_x)
# run the test harness for evaluating a model

def run_test_harness(model):

    # fit model

    

    history = model.fit(train_x, train_y, epochs=15, batch_size=64, validation_data=(val_x, val_y))

    # evaluate model

    _, acc = model.evaluate(test_x, test_y)

    print("Test accuracy" , end = ' ')

    print('> %.3f' % (acc * 100.0))

    # learning curves

    summarize_diagnostics(history)
def model1():

    model = Sequential()

    model.add(Conv2D(32, (5, 5), activation='relu',input_shape=(32, 32, 3)))

    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model

run_test_harness(model1())
def model2():

    model = Sequential()

    model.add(Conv2D(32, (5, 5), activation='relu' ,input_shape=(32, 32, 3)))

    model.add(MaxPooling2D(3,3))

    model.add(Conv2D(64, (5, 5), activation='relu' ))

    model.add(MaxPooling2D((3, 3)))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu' ))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model

run_test_harness(model2())
def model3():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu' ,  input_shape=(32, 32, 3)))

    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3, 3), activation='relu' ))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu' ))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model

run_test_harness(model3())
def model4():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu',  input_shape=(32, 32, 3)))

    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu' ))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu' ))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model

run_test_harness(model4())
def model5():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), strides = 2 , activation='relu' ,  input_shape=(32, 32, 3)))

    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3, 3),strides = 2 ,activation='relu' ))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu' ))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model

run_test_harness(model5())
def model6():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding = 'same' ,activation='relu',  input_shape=(32, 32, 3)))

    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3, 3), padding = 'same' ,  activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding = 'same' , activation='relu' ))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu' ))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model

run_test_harness(model6())
def model7():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu' , strides=  2 , padding = 'same' ,  input_shape=(32, 32, 3)))

    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3, 3), activation='relu' , strides = 2 , padding= 'same'))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu' ))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model

run_test_harness(model7())
def model8():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding = 'same' ,activation='tanh',  input_shape=(32, 32, 3)))

    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3, 3), padding = 'same' ,  activation='tanh'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding = 'same' , activation='tanh' ))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu' ))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model

run_test_harness(model8())
def model9():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding = 'same' ,activation='sigmoid',  input_shape=(32, 32, 3)))

    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3, 3), padding = 'same' ,  activation='sigmoid'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding = 'same' , activation='sigmoid' ))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu' ))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model

run_test_harness(model9())