#Import Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image
from IPython.core.display import HTML 
import matplotlib.pyplot as plt
%matplotlib inline
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
import os
print(os.listdir("../input"))
test_data = np.array(pd.read_csv(r'/kaggle/input/fashionmnist/fashion-mnist_test.csv'), dtype='float32')
embed_count = 2500
x_test = test_data[:embed_count, 1:] / 255
y_test = test_data[:embed_count, 0]