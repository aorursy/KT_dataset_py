import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

import random

from random import shuffle

from PIL import Image

print(tf.__version__)



def load_data(path):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)



(x_train, y_train), (x_test, y_test) = load_data('../input/mnist.npz')
IMG_SIZE = 28

LR = 0.0001

EPOCH = 10



MODEL_NAME = 'MNIST-{}-{}-{}.model'.format(LR, '2conv-'+str(IMG_SIZE)+'x'+str(IMG_SIZE), 'EPOCH-'+str(EPOCH)) # just so we remember which saved model is which, sizes must match
print(y_train.shape[0])

train_data = []

test_data = []

for num in range(x_train.shape[0]):

    

    # onehot

    y_train_onehot = np.eye(10)[y_train[num].reshape(-1)]

    train_data.append([x_train[num].reshape(IMG_SIZE, IMG_SIZE, 1), y_train_onehot])



shuffle(train_data)

    

for numb in range(x_test.shape[0]):

    

    y_test_onehot = np.eye(10)[y_test[numb].reshape(-1)]

    test_data.append([x_test[numb].reshape(IMG_SIZE, IMG_SIZE, 1), y_test_onehot])

    

shuffle(test_data)

print(np.array(train_data).shape)

print(np.array(test_data).shape)



print(y_train_onehot.shape)

print(y_test_onehot.shape)
import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.estimator import regression



convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')



convnet = conv_2d(convnet, 32, 5, activation='relu')

convnet = max_pool_2d(convnet, 5)



convnet = conv_2d(convnet, 64, 5, activation='relu')

convnet = max_pool_2d(convnet, 5)



convnet = fully_connected(convnet, 1024, activation='relu')

convnet = dropout(convnet, 0.8)



convnet = fully_connected(convnet, 10, activation='softmax')

convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')



model = tflearn.DNN(convnet, tensorboard_dir='log')
X = np.array([i[0] for i in train_data])

Y = np.array([i[1] for i in train_data]).reshape(y_train.shape[0], 10)

print(np.array(Y[1]).reshape(1,10))

print(Y.shape)

# print(np.array(x_train_data[0]).shape)

# # img = Image.fromarray(np.array(x_train_data[0]))

# assert x_train_data[0][0].all()==x_train[0].all()

# print(x_train_data[0][0])

# img = Image.fromarray(X[0])

plt.imshow(X[1].reshape(28,28),'gray')



test_x = np.array([i[0] for i in test_data])

test_y = np.array([i[1] for i in test_data]).reshape(y_test.shape[0], 10)
model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCH, validation_set=({'input': test_x}, {'targets': test_y}),

    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
import matplotlib.pyplot as plt



fig = plt.figure()



for num, data in enumerate(x_test[100:142]):

    

    y = fig.add_subplot(6, 7, num + 1)

#     print(data.shape)

    orig = data.reshape(IMG_SIZE, IMG_SIZE, 1)

#     print(orig.shape)

    # model_out = model.predict([data])[0]

    model_out = model.predict([orig])[0]

    str_label = str(np.argmax(model_out))

    

    y.imshow(data)

    plt.title(str_label)

    y.axes.get_xaxis().set_visible(False)

    y.axes.get_yaxis().set_visible(False)

    plt.savefig(MODEL_NAME + '.jpg', dpi=400)

plt.show()