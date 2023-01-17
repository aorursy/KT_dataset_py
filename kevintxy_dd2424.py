# Contains some adjustments and tests that better suit running on a 6gb GPU
#
import pickle
import tensorflow as tf
import gc
from tensorflow.keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


class AlexNet(models.Sequential):
    def __init__(self, input_shape, classes):
        super().__init__()
        self.add(layers.Conv2D(filters=96, kernel_size=(11, 11),
                               strides=(4, 4), padding='valid',
                               activation='relu',
                               input_shape=input_shape))
        # Local Normalization layer
        self.add(LRN())
        self.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='valid'))

        self.add(layers.Conv2D(filters=256, kernel_size=(5, 5),
                               strides=(1, 1), padding='same',
                               activation='relu'))
        self.add(LRN())
        self.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='valid'))

        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                               strides=(1, 1), padding='same',
                               activation='relu'))
        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                               strides=(1, 1), padding='same',
                               activation='relu'))
        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                               strides=(1, 1), padding='same',
                               activation='relu'))
        self.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='valid'))

        self.add(layers.Flatten())
        self.add(layers.Dense(4096, activation='relu'))
        self.add(layers.Dropout(.5))
        self.add(layers.Dense(4096, activation='relu'))
        self.add(layers.Dropout(.5))
        self.add(layers.Dense(classes, activation='softmax'))


class LRN(layers.Layer):
    """
    custom layer, used to implement local response normalization,
    as defined in the original alexnet paper
    """
    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        return tf.nn.local_response_normalization(
            x, depth_radius=self.n, bias=self.k,
            alpha=self.alpha, beta=self.beta)

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def loadBatch(filename):
    path = '/kaggle/input/'
    with open(path + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        y = np.asarray(dict[b'labels'])
        X = dict[b'data']
        X = X / 255.0
        Y = to_categorical(y)
    return X, Y


def normalize(X, mean_train, std_train):
    return (X - mean_train) / std_train


def resize(X_cifar):
    """
    rescale a cifar image to get an image of size 227*227*3
    """
    X_cifar = X_cifar.reshape(np.size(X_cifar, 0), 3, 32, 32).transpose(0, 2, 3, 1)
    return tf.image.resize(X_cifar, size = [227, 227])


def main():

    # Load data
    Xtrain, Ytrain = loadBatch('data_batch_1')
    Xval, Yval = loadBatch('data_batch_2')

    lim = 8000
    # clip everything to run/check faster
    Xtrain = Xtrain[:lim, :]
    Ytrain = Ytrain[:lim, :]
    Xval = Xval[:lim, :]
    Yval = Yval[:lim, :]

    # normalize the data
    mean_train = np.mean(Xtrain, axis=0)
    std_train = np.std(Xtrain, axis=0)
    Xtrain = normalize(Xtrain, mean_train, std_train)
    Xval = normalize(Xval, mean_train, std_train)

    # reshape cifar images from 32 * 32 * 3 to 227 * 227 * 3
    # not a very good solution
    Xtrain = resize(Xtrain)
    Xval = resize(Xval)

    # hyper-params
    eta = 0.01
    n_batch = 128
    epochs = 15

    # create & train the model
    model = AlexNet((227, 227, 3), 10)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=eta, momentum=.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    history = model.fit(Xtrain, Ytrain, epochs=epochs,
                        batch_size=n_batch,
                        validation_data=(Xval, Yval))
    
    
    
    gc.collect()

    # plots
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('result.png')

    del Xtrain
    del Ytrain
    del Xval
    del Yval
    Xtest, Ytest = loadBatch('test_batch')
    Xtest = Xtest[:2000, :]
    Ytest = Ytest[:2000, :]
    Xtest = normalize(Xtest, mean_train, std_train)
    Xtest = resize(Xtest)
    test_loss, test_acc = model.evaluate(Xtest, Ytest, verbose=2)
    


if __name__ == '__main__':
    main()
