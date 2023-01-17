# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.utils.np_utils import to_categorical

from keras.layers import Activation, Dense



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train_labels = train["label"]



# Drop 'label' column

train_images = train.drop(labels = ["label"],axis = 1) 



# # free some space

# del train 



g = sns.countplot(train_labels)



class_name = np.unique(train_labels)

print(class_name)
train_images = train_images / 255.0



test_images = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

train_images = train_images.values.reshape(-1,28,28)

test = test.values.reshape(-1,28,28)
test_images = test_images.values.reshape(-1,28,28)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

#train_labels = to_categorical(train_labels, num_classes = 10)
g = plt.imshow(train_images[0][:,:])
input_shape = train_images.shape

print(train_images[0].shape)

print(train_labels.shape)
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
prediction = model.predict(test_images)
print(np.argmax(prediction[0]))
plt.bar(np.arange(len(prediction[4])),prediction[4])
predict_values = []

for predict in prediction:

    (predict_values.append(np.argmax(predict)))

    

ImageId= list(range(1,len(prediction)+1))

print(len(predict_values),len(ImageId))
submission = pd.DataFrame({

        "ImageId": ImageId,

        "Label": predict_values

    })
submission.to_csv('MNIST_Digit_Recognizer.csv',index=False)

display(submission.tail())

print("The shape of the sumbission file is {}".format(submission.shape))


# Function to save result to a file

def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": np.argmax(preds)}).to_csv(fname, index=False, header=True)
# Write to file your test score for submission

write_preds(predict_values, "keras_kaggle_conv.csv")
submission.head()