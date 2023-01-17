%matplotlib inline

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

from sklearn.metrics import confusion_matrix
tf.__version__
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("www.yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", one_hot=True)