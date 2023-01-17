from keras.callbacks import Callback

from IPython import display

import pylab



# setup callbacks

class PlottingCallback(Callback):

    def __init__(self, keys=['loss', 'val_loss'], realtime_display=False, clear=True):

        self.epoch = []

        self.logsDict={}

        self.keys = keys

        self.clear = clear

        self.realtime_display = realtime_display

        for k in keys:

            self.logsDict[k] = []

        

    def on_epoch_end(self,epoch, logs={}):

        print('epoch '+ str(epoch))

        print(logs)

        self.epoch.append(epoch)

        for k in self.keys:

            if k in logs:

                self.logsDict[k].append(logs[k])

            else:

                self.logsDict[k].append(None)

        if self.realtime_display:

            plt.clf()

            for k in self.keys:

                plt.plot(self.epoch,self.logsDict[k])

            plt.legend(self.keys)

            display.display(pylab.gcf())

            if self.clear:

                display.clear_output(wait=True)



    def plot(self):

        plt.figure()

        for k in self.keys:

            plt.plot(self.epoch,self.logsDict[k])

        plt.legend(self.keys)

        

plotting_callback_loss = PlottingCallback(['loss', 'val_loss']) 

plotting_callback_acc = PlottingCallback(['acc', 'val_acc'])
import gzip, pickle, sys

f = gzip.open('../input/mnist.pkl.gz', 'rb')

if sys.version_info < (3,):

    (X_train, y_train), (X_test, y_test) = pickle.load(f)

else:

    (X_train, y_train), (X_test, y_test) = pickle.load(f, encoding="bytes")

    

print(X_train.shape)

print(y_train.shape)
import matplotlib.pyplot as plt



%matplotlib inline



for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(X_train[i], cmap='gray', interpolation='nearest')

    plt.title("class={}".format(y_train[i]))

    plt.axis('off')
X_train = X_train.reshape(60000, 784)

X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255

print("Training matrix shape", X_train.shape)

print("Testing matrix shape", X_test.shape)
print(y_train[1])

print(y_train[2])
from keras.utils import np_utils



Y_train = np_utils.to_categorical(y_train, 10)

Y_test = np_utils.to_categorical(y_test, 10)
print(Y_train[1])

print(Y_train[2])
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation



model = Sequential()

model.add(Dense(512, input_shape=(784,)))

model.add(Activation('relu')) 

model.add(Dropout(0.2))   



model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.2))   



model.add(Dense(10))

model.add(Activation('softmax')) 
## Compile + specify loss and optimizer
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, Y_train,

          batch_size=128, nb_epoch=10,

          verbose=0,

          validation_data=(X_test, Y_test),

          callbacks=[plotting_callback_loss, plotting_callback_acc])
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])
plotting_callback_loss.plot()

plotting_callback_acc.plot()
predicted_classes = model.predict_classes(X_test, verbose=0)



import numpy as np



correct_indices = np.nonzero(predicted_classes == y_test)[0]

incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
for i, correct in enumerate(correct_indices[:4]):

    plt.subplot(2,2,i+1)

    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("predicted={}, true={}".format(predicted_classes[correct], y_test[correct]))

    plt.axis("off")
for i, incorrect in enumerate(incorrect_indices[:4]):

    plt.subplot(2,2,i+1)

    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted={}, True={}".format(predicted_classes[incorrect], y_test[incorrect]))

    plt.axis("off")