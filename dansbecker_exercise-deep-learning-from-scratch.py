import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras



img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"

fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')

x, y = prep_data(fashion_data)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_7 import *

print("Setup Complete")
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



# Your Code Here

____



# Check your answer

q_1.check()
#q_1.solution()
# Your code here

____



# Check your answer

q_2.check()
# q_2.hint()

#q_2.solution()
# Your code here

____



# Check your answer

q_3.check()
# q_3.solution()
# Your code to compile the model in this cell

____



# Check your answer

q_4.check()
# q_4.solution()
# Your code to fit the model here

____



# Check your answer

q_5.check()
#q_5.solution()
# Your code below

____





# Don't remove this line (ensures comptibility with tensorflow 2.0)

second_fashion_model.history.history['val_acc'] = second_fashion_model.history.history['val_accuracy']

# Check your answer

q_6.check()
#q_6.solution()