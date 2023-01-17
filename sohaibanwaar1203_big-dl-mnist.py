!pip install pyspark
!pip install bigdl

import matplotlib
matplotlib.use('Agg')
%pylab inline
import pandas
import datetime as dt

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
import matplotlib.pyplot as plt
from pyspark import SparkContext
from matplotlib.pyplot import imshow

sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[4]").set("spark.driver.memory","2g"))
init_engine()
!pip install get_mnist


from bigdl.dataset import mnist
from bigdl.util.common import *

mnist_path = "datasets/mnist"
(X_train, Y_train), (X_test, Y_test) = mnist.load_data(mnist_path)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
type(X_train)
# Get and store MNIST into RDD of Sample, please edit the "mnist_path" accordingly.

from __future__ import print_function
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import *
num_fc = 512
num_outputs = 10
model = Sequential()
model.add(Reshape((1, 28, 28), input_shape=(28, 28, 1)))
model.add(Convolution2D(20, 3, 3, activation="relu", input_shape=(1, 28, 28)))
model.add(MaxPooling2D())
model.add(Convolution2D(50, 3, 3, activation="relu", name="conv2_5x5"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(num_fc, activation="relu", name="fc1"))
model.add(Dense(num_outputs, activation="softmax", name="fc2"))
from bigdl.nn.criterion import *

model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=8, nb_epoch=1,
validation_data=(X_test, Y_test))
print(model.evaluate(X_test,Y_test))
a=model.evaluate(X_test,Y_test)
print(*a, sep='\n')
