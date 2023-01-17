import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from parameterizedcnn import ParameterizedCNN 



from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.utils import to_categorical



import tensorflow as tf

import sys
mtrain = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

mtest = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = np.array(mtrain)

mnist_test = np.array(mtest)

mnist_train_label = mnist_train[:,0]

mnist_test_label = mnist_test[:,0]

mnist_train = mnist_train[:,1:]

mnist_test = mnist_test[:,1:]

mnist_train=mnist_train.reshape(60000,28,28,1)

mnist_train = mnist_train.astype("float32")/255.0

mnist_test=mnist_test.reshape(10000,28,28,1)

mnist_test = mnist_test.astype("float32")/255.0





plt.imshow(mnist_train[0].reshape(28,28), cmap='gray')

mnist_train_label = to_categorical(mnist_train_label)

mnist_test_label = to_categorical(mnist_test_label)
def step_function(X,y):

    with tf.GradientTape() as tape:

        pred = model(X)

        loss = categorical_crossentropy(y, pred)

        

        grads=tape.gradient(loss, model.trainable_variables)

        opt.apply_gradients(zip(grads, model.trainable_variables))
EPOCHS = 25

BS = 64

INIT_LR = 1e-3



# model parameters

default_parameters = {"filters":[16,32,64], "filter_size":[3,3,3], "pool_size":[2,2,2],"padding": ["same","same","same"], "drop_out":[0.3,0.4,0.5],"dense":256}

img_shape = (28,28,1)

mnist_classes = 10



model = ParameterizedCNN.generate_model(input_shape=img_shape, hyperparameters = default_parameters, classes=mnist_classes)



opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)

model.summary()
updates = int(len(mnist_train)/BS)

for i in range(0, updates):

    start = i * BS

    end = start + BS

    

    step_function(mnist_train[start:end], mnist_train_label[start:end])
model.compile(optimizer=opt, loss=categorical_crossentropy,	metrics=["acc"])



(loss, acc) = model.evaluate(mnist_test, mnist_test_label)

print("Model accuracy : {:.4f}".format(acc))