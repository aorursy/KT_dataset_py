from google.colab import drive

drive.mount('/content/drive/')
import io

from google.colab import files



datapath = '/content/drive/My Drive/ConeEstimation/'
from numpy.random import seed

seed(1029)

from tensorflow import set_random_seed

set_random_seed(1029)



import numpy as np

import pandas as pd

from imageio import imread

from skimage.transform import resize

import os

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
import numpy as np

import pandas as pd

from imageio import imread

from skimage.transform import resize

import os



df1 = pd.read_csv(datapath + 'training.csv')

df2 = pd.read_csv(datapath + 'testing.csv')
train = df1



newidTrain = [str(i) for i in train['Id'][0:900]]

width, height = 1920/6, 1080/6

file = os.listdir()



training_images1 = [imread(datapath + 'TrainingImages/' + j) for j in newidTrain]

resized = [resize(i, (width, height)) for i in training_images1]

training_images1 = np.array(resized)
newidTrain = [str(i) for i in train['Id'][901:1800]]

width, height = 1920/6, 1080/6

file = os.listdir()



training_images2 = [imread(datapath + 'TrainingImages/' + j) for j in newidTrain]

resized = [resize(i, (width, height)) for i in training_images2]

training_images2 = np.array(resized)
newidTrain = [str(i) for i in train['Id'][1801:]]

width, height = 1920/6, 1080/6

file = os.listdir()



training_images3 = [imread(datapath + 'TrainingImages/' + j) for j in newidTrain]

resized = [resize(i, (width, height)) for i in training_images3]

training_images3 = np.array(resized)
training_images = np.concatenate((training_images1, training_images2, training_images3))
test = df2



newidTest = [str(i) for i in test['Id']]

width, height = 1920/6, 1080/6

file = os.listdir()



testing_images = [imread(datapath + 'TestingImages/' + j) for j in newidTest]

resized = [resize(i, (width, height)) for i in testing_images]

testing_images = np.array(resized)

import numpy as np

np.random.seed(1029)

import pandas as pd

from imageio import imread

from skimage.transform import resize

import os

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical

from keras.datasets import mnist

import matplotlib.pyplot as plt

import pandas as pd

from tensorflow.keras import datasets, layers, models

from sklearn.model_selection import train_test_split

x_train = training_images

y3_train = train["Distance"][0:2812]

y3_train = np.array(y3_train)



from keras.layers.convolutional import Conv2D, MaxPooling2D
#Modelling a Sequential Model

from keras.layers import Dense, Dropout, Flatten

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Conv2D(16, kernel_size=(7, 7), activation='relu', input_shape=(320, 180, 3)))

model.add(tf.keras.layers.Conv2D(16, kernel_size=(7, 7), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))



model.add(tf.keras.layers.Conv2D(8, kernel_size=(5, 5), activation='relu'))

model.add(tf.keras.layers.Conv2D(8, kernel_size=(5, 5), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))



model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())



model.add(tf.keras.layers.Dense(16, activation = 'relu'))

model.add(tf.keras.layers.Dense(8, activation = 'relu'))



model.add(tf.keras.layers.Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error') 



model.fit(x_train, y3_train, epochs = 150, batch_size = 32, validation_split = 0.05)



#Predicting

predictions = model.predict(testing_images)

print(predictions)
model.summary()
arr = np.array(predictions)

df_solution = pd.read_csv(datapath + 'testing.csv')



df_solution['Distance'] = arr

df_solution.to_csv(datapath + 'abhi4.csv', index = False)