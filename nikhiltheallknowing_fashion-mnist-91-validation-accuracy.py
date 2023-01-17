import pandas as pd

import numpy as np

import pylab as plt



from IPython.display import clear_output



from keras.callbacks import Callback

from keras.models import Sequential

from keras.utils import to_categorical

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten
train_data = pd.read_csv('../input/fashion-mnist_train.csv')

test_data = pd.read_csv('../input/fashion-mnist_test.csv')
y = to_categorical(train_data.label.values)

X = np.expand_dims(train_data.drop(['label'], axis=1).values.reshape((-1,28,28)), -1)/255



y_ = to_categorical(test_data.label.values)

X_ = np.expand_dims(test_data.drop(['label'], axis=1).values.reshape((-1,28,28)), -1)/255
class PlotLearning(Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        

        self.logs = []

        



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        

        clear_output(wait=True)

        

        ax1.set_yscale('Log')

        ax1.plot(self.x, self.losses, label="loss")

        ax1.plot(self.x, self.val_losses, label="val_loss")

        ax1.legend()

        

        ax2.plot(self.x, self.acc, label="accuracy", c='b')

        ax2.scatter(self.x, self.val_acc, label="validation accuracy", c='r')

        ax2.legend()

        

        plt.show();

        

        

plot = PlotLearning()
model = Sequential([

    Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1), kernel_regularizer='l2'),

    BatchNormalization(),

    Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1), kernel_regularizer='l2'),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    

    Conv2D(16, (3,3), activation='relu', kernel_regularizer='l2'),

    BatchNormalization(),

    Conv2D(16, (3,3), activation='relu', kernel_regularizer='l2'),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    

    Flatten(),

    

    Dense(10, activation='softmax')

])



model.compile(

    optimizer='Adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)



model.summary()
model.fit(X, y, epochs=100, batch_size=256, validation_data=(X_, y_), callbacks=[plot], shuffle=True)
model.save('best_model_ever - fashionMNIST.h5')