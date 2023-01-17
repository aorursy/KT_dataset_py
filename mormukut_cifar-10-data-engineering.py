from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
from keras.utils import np_utils

# load data
# 1 a)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train)
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
print(Y_train)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# 2) a)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
#  so dataset has 10 classes. Total images 60000. training set contain 50000 and tesset contain 10000
# 32*32 is the dimension of image in this dataset

