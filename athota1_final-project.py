import keras

from keras.models import Sequential

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import cifar10

from keras import regularizers, optimizers

import numpy as np

import cv2



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



#z-score

mean = np.mean(x_train,axis=(0,1,2,3))

std = np.std(x_train,axis=(0,1,2,3))

x_train = (x_train-mean)/(std+1e-7)

x_test = (x_test-mean)/(std+1e-7)



num_classes = 10

y_train = np_utils.to_categorical(y_train,num_classes)

y_test = np_utils.to_categorical(y_test,num_classes)



baseMapNum = 32

weight_decay = 1e-4

model = Sequential()

model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))



model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))



model.summary()



#data augmentation

datagen = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    vertical_flip=False

    )

datagen.fit(x_train)



#training

batch_size = 64

epochs=10

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

model.compile(loss='categorical_crossentropy',

        optimizer=opt_rms,

        metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=3*epochs,verbose=1,validation_data=(x_test,y_test))

model.save_weights('cifar10_normal_rms_ep75.h5')

'''

opt_rms = keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)

model.compile(loss='categorical_crossentropy',

        optimizer=opt_rms,

        metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

model.save_weights('cifar10_normal_rms_ep100.h5')



opt_rms = keras.optimizers.rmsprop(lr=0.0003,decay=1e-6)

model.compile(loss='categorical_crossentropy',

        optimizer=opt_rms,

        metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

model.save_weights('cifar10_normal_rms_ep125.h5')

'''

#testing - no kaggle eval

scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)

print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
import keras

from keras.models import Sequential

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import cifar100

from keras import regularizers, optimizers

import numpy as np

import cv2



(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



#z-score

mean = np.mean(x_train,axis=(0,1,2,3))

std = np.std(x_train,axis=(0,1,2,3))

x_train = (x_train-mean)/(std+1e-7)

x_test = (x_test-mean)/(std+1e-7)



num_classes = 100

y_train = np_utils.to_categorical(y_train,num_classes)

y_test = np_utils.to_categorical(y_test,num_classes)



baseMapNum = 32

weight_decay = 1e-4

model = Sequential()

model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))



model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))



model.summary()



#data augmentation

datagen = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    vertical_flip=False

    )

datagen.fit(x_train)



#training

batch_size = 64

epochs=10

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

model.compile(loss='categorical_crossentropy',

        optimizer=opt_rms,

        metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=3*epochs,verbose=1,validation_data=(x_test,y_test))

model.save_weights('cifar10_normal_rms_ep75.h5')

'''

opt_rms = keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)

model.compile(loss='categorical_crossentropy',

        optimizer=opt_rms,

        metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

model.save_weights('cifar10_normal_rms_ep100.h5')



opt_rms = keras.optimizers.rmsprop(lr=0.0003,decay=1e-6)

model.compile(loss='categorical_crossentropy',

        optimizer=opt_rms,

        metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

model.save_weights('cifar10_normal_rms_ep125.h5')

'''

#testing - no kaggle eval

scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)

print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
import keras

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras import optimizers

import numpy as np

from keras.layers.core import Lambda

from keras import backend as K

from keras import regularizers



class cifar10vgg:

    def __init__(self,train=True):

        self.num_classes = 10

        self.weight_decay = 0.0005

        self.x_shape = [32,32,3]



        self.model = self.build_model()

        if train:

            self.model = self.train(self.model)

        else:

            self.model.load_weights('cifar10vgg.h5')





    def build_model(self):

        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.



        model = Sequential()

        weight_decay = self.weight_decay



        model.add(Conv2D(64, (3, 3), padding='same',

                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.3))



        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))



        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))



        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))





        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))





        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.5))



        model.add(Flatten())

        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes))

        model.add(Activation('softmax'))

        return model





    def normalize(self,X_train,X_test):

        #this function normalize inputs for zero mean and unit variance

        # it is used when training a model.

        # Input: training set and test set

        # Output: normalized training set and test set according to the trianing set statistics.

        mean = np.mean(X_train,axis=(0,1,2,3))

        std = np.std(X_train, axis=(0, 1, 2, 3))

        X_train = (X_train-mean)/(std+1e-7)

        X_test = (X_test-mean)/(std+1e-7)

        return X_train, X_test



    def normalize_production(self,x):

        #this function is used to normalize instances in production according to saved training set statistics

        # Input: X - a training set

        # Output X - a normalized training set according to normalization constants.



        #these values produced during first training and are general for the standard cifar10 training set normalization

        mean = 120.707

        std = 64.15

        return (x-mean)/(std+1e-7)



    def predict(self,x,normalize=True,batch_size=50):

        if normalize:

            x = self.normalize_production(x)

        return self.model.predict(x,batch_size)



    def train(self,model):



        #training parameters

        batch_size = 128

        maxepoches = 250

        learning_rate = 0.1

        lr_decay = 1e-6

        lr_drop = 20

        # The data, shuffled and split between train and test sets:

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype('float32')

        x_test = x_test.astype('float32')

        x_train, x_test = self.normalize(x_train, x_test)



        y_train = keras.utils.to_categorical(y_train, self.num_classes)

        y_test = keras.utils.to_categorical(y_test, self.num_classes)



        def lr_scheduler(epoch):

            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)



        #data augmentation

        datagen = ImageDataGenerator(

            featurewise_center=False,  # set input mean to 0 over the dataset

            samplewise_center=False,  # set each sample mean to 0

            featurewise_std_normalization=False,  # divide inputs by std of the dataset

            samplewise_std_normalization=False,  # divide each input by its std

            zca_whitening=False,  # apply ZCA whitening

            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

            horizontal_flip=True,  # randomly flip images

            vertical_flip=False)  # randomly flip images

        # (std, mean, and principal components if ZCA whitening is applied).

        datagen.fit(x_train)







        #optimization details

        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])





        # training process in a for loop with learning rate drop every 25 epoches.



        historytemp = model.fit_generator(datagen.flow(x_train, y_train,

                                         batch_size=batch_size),

                            steps_per_epoch=x_train.shape[0] // batch_size,

                            epochs=maxepoches,

                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)

        model.save_weights('cifar10vgg.h5')

        return model



if __name__ == '__main__':





    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')

    x_test = x_test.astype('float32')



    y_train = keras.utils.to_categorical(y_train, 10)

    y_test = keras.utils.to_categorical(y_test, 10)



    model = cifar10vgg()



    predicted_x = model.predict(x_test)

    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)



    loss = sum(residuals)/len(residuals)

    print("the validation 0/1 loss is: ",loss)
import keras

from keras.datasets import cifar100

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras import optimizers

import numpy as np

from keras.layers.core import Lambda

from keras import backend as K

from keras import regularizers



class cifar100vgg:

    def __init__(self,train=True):

        self.num_classes = 100

        self.weight_decay = 0.0005

        self.x_shape = [32,32,3]



        self.model = self.build_model()

        if train:

            self.model = self.train(self.model)

        else:

            self.model.load_weights('cifar100vgg.h5')





    def build_model(self):

        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.



        model = Sequential()

        weight_decay = self.weight_decay



        model.add(Conv2D(64, (3, 3), padding='same',

                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.3))



        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))



        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))



        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))





        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))





        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.4))



        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.5))



        model.add(Flatten())

        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())



        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes))

        model.add(Activation('softmax'))

        return model





    def normalize(self,X_train,X_test):

        #this function normalize inputs for zero mean and unit variance

        # it is used when training a model.

        # Input: training set and test set

        # Output: normalized training set and test set according to the trianing set statistics.

        mean = np.mean(X_train,axis=(0,1,2,3))

        std = np.std(X_train, axis=(0, 1, 2, 3))

        print(mean)

        print(std)

        X_train = (X_train-mean)/(std+1e-7)

        X_test = (X_test-mean)/(std+1e-7)

        return X_train, X_test



    def normalize_production(self,x):

        #this function is used to normalize instances in production according to saved training set statistics

        # Input: X - a training set

        # Output X - a normalized training set according to normalization constants.



        #these values produced during first training and are general for the standard cifar10 training set normalization

        mean = 121.936

        std = 68.389

        return (x-mean)/(std+1e-7)



    def predict(self,x,normalize=True,batch_size=50):

        if normalize:

            x = self.normalize_production(x)

        return self.model.predict(x,batch_size)



    def train(self,model):



        #training parameters

        batch_size = 128

        maxepoches = 250

        learning_rate = 0.1

        lr_decay = 1e-6

        lr_drop = 20



        # The data, shuffled and split between train and test sets:

        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        x_train = x_train.astype('float32')

        x_test = x_test.astype('float32')

        x_train, x_test = self.normalize(x_train, x_test)



        y_train = keras.utils.to_categorical(y_train, self.num_classes)

        y_test = keras.utils.to_categorical(y_test, self.num_classes)





        def lr_scheduler(epoch):

            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)





        #data augmentation

        datagen = ImageDataGenerator(

            featurewise_center=False,  # set input mean to 0 over the dataset

            samplewise_center=False,  # set each sample mean to 0

            featurewise_std_normalization=False,  # divide inputs by std of the dataset

            samplewise_std_normalization=False,  # divide each input by its std

            zca_whitening=False,  # apply ZCA whitening

            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

            horizontal_flip=True,  # randomly flip images

            vertical_flip=False)  # randomly flip images

        # (std, mean, and principal components if ZCA whitening is applied).

        datagen.fit(x_train)







        #optimization details

        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])





        # training process in a for loop with learning rate drop every 25 epoches.



        historytemp = model.fit_generator(datagen.flow(x_train, y_train,

                                         batch_size=batch_size),

                            steps_per_epoch=x_train.shape[0] // batch_size,

                            epochs=maxepoches,

                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)

        model.save_weights('cifar100vgg.h5')

        return model



if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.astype('float32')

    x_test = x_test.astype('float32')



    y_train = keras.utils.to_categorical(y_train, 100)

    y_test = keras.utils.to_categorical(y_test, 100)



    model = cifar100vgg()



    predicted_x = model.predict(x_test)

    residuals = (np.argmax(predicted_x,1)!=np.argmax(y_test,1))

    loss = sum(residuals)/len(residuals)

    print("the validation 0/1 loss is: ",loss)