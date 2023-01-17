from google.colab import drive

drive.mount('/content/drive')
import io

from google.colab import files



datapath = '/content/drive/My Drive/'
import numpy as np

import pandas as pd

from imageio import imread

from skimage.transform import resize

import os



df1 = pd.read_csv(datapath + 'training.csv')

df2 = pd.read_csv(datapath + 'testing.csv')
train = df1



newidTrain = [str(i) for i in train['Id']]

width, height = 512,512

file = os.listdir()



training_images = [imread(datapath + 'Images/Images/Training/' + j) for j in newidTrain]

resized = [resize(i, (width, height)) for i in training_images]

training_images = np.array(resized)
test = df2



newidTest = [str(i) for i in test['Id']]

width, height = 512,512

file = os.listdir()



testing_images = [imread(datapath + 'Images/Images/Testing/' + j) for j in newidTest]

resized = [resize(i, (width, height)) for i in testing_images]

testing_images = np.array(resized)
import tensorflow as tf

import numpy as np



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



y_train = train["Category"]

y_train = np.array(y_train)



x_test = testing_images
# I have the model saved on my Drive and can share it if required

model = tf.keras.models.load_model(datapath + 'driveattempt1.model')
model.get_config()
model.summary()
predictions = model.predict(x_test)

print(predictions)
pred = np.around(predictions)
print(pred)
df_solution = pd.read_csv(datapath + 'sample.csv')
df_solution['Category'] = pred

df_solution.to_csv(datapath + 'abhi_drive_solF1.csv', index = False)