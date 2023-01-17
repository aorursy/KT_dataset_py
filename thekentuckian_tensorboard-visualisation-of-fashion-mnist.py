#Import Numpy for statistical calculations
import numpy as np
# Import Pandas for data manipulation using dataframes
import pandas as pd
# Import Warnings 
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image
from IPython.core.display import HTML 
# Import matplotlib Library for data visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
test_data = np.array(pd.read_csv(r'/kaggle/input/fashion-mnist_test.csv'), dtype='float32')
embed_count = 2500
x_test = test_data[:embed_count, 1:] / 255
y_test = test_data[:embed_count, 0]
