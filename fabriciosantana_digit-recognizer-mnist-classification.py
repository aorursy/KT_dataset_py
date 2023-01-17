# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals



import tensorflow as tf



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tensorflow import keras



print(tf.__version__)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

fashion_mnist_train_dataset = pd.read_csv ('../input/train.csv', sep=',')

fashion_mnist_test_dataset = pd.read_csv ('../input/test.csv', sep=',')

fashion_mnist_train_dataset = fashion_mnist_train_dataset.values

fashion_mnist_test_dataset = fashion_mnist_test_dataset.values

print(fashion_mnist_train_dataset.shape)

print(fashion_mnist_test_dataset.shape)
fashion_mnist_train_dataset_labes = fashion_mnist_train_dataset[0:,0]

#fashion_mnist_test_dataset_labes = fashion_mnist_test_dataset[0:,0]

print(fashion_mnist_train_dataset_labes)

#print(fashion_mnist_test_dataset_labes)

print(fashion_mnist_train_dataset_labes.shape)

#print(fashion_mnist_test_dataset_labes.shape)
fashion_mnist_train_dataset_images = fashion_mnist_train_dataset[0:,1:]

fashion_mnist_test_dataset_images = fashion_mnist_test_dataset[0:,0:]

print(fashion_mnist_train_dataset_images.shape)

print(fashion_mnist_test_dataset_images.shape)
fashion_mnist_train_dataset_images = fashion_mnist_train_dataset_images.reshape(840, 50, 28,28)

fashion_mnist_train_dataset_images = fashion_mnist_train_dataset_images.reshape(42000, 28,28)

fashion_mnist_test_dataset_images = fashion_mnist_test_dataset_images.reshape(560, 50, 28,28)

fashion_mnist_test_dataset_images = fashion_mnist_test_dataset_images.reshape(28000, 28,28)

print(fashion_mnist_train_dataset_images.shape)

print(fashion_mnist_test_dataset_images.shape)
image_sample_index = 3

pixel_index = 3

train_image_sample = fashion_mnist_train_dataset_images[image_sample_index]

train_image_sample_pixel = fashion_mnist_train_dataset_images[image_sample_index][0][pixel_index]

train_label_sample = fashion_mnist_train_dataset_labes[image_sample_index]

print(fashion_mnist_train_dataset_labes)

print(fashion_mnist_train_dataset_labes.shape)

#print(fashion_mnist_train_dataset_images)

print(train_image_sample_pixel)

print("{}:{}".format(train_label_sample, class_names[train_label_sample]))
plt.figure()

plt.imshow(train_image_sample)

plt.colorbar()

plt.grid(False)

plt.show()
keras.__version__
validation_fashion_mnist_train_dataset_images = fashion_mnist_train_dataset_images[:30000]

partial_fashion_mnist_train_dataset_images = fashion_mnist_train_dataset_images[30000:]



validation_fashion_mnist_train_dataset_labes = fashion_mnist_train_dataset_labes[:30000]

partial_fashion_mnist_train_dataset_labes = fashion_mnist_train_dataset_labes[30000:]



model = keras.Sequential([

            keras.layers.Flatten(input_shape=(28, 28)),

            keras.layers.Dense(128, activation=tf.nn.relu),

            keras.layers.Dense(10, activation=tf.nn.softmax)

        ])



model.compile(optimizer='adam',

                      loss='sparse_categorical_crossentropy',

                      metrics=['accuracy'])



model.summary()
history = model.fit(fashion_mnist_train_dataset_images, fashion_mnist_train_dataset_labes, 

                    epochs=5

                    ,batch_size=512, 

                    validation_data=(validation_fashion_mnist_train_dataset_images,validation_fashion_mnist_train_dataset_labes))
history_dict = history.history

print(history_dict.keys())
acc = history_dict['acc']

val_acc = history_dict['val_acc']

loss = history_dict['loss']

val_loss = history_dict['val_loss']



epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
predictions = model.predict(fashion_mnist_test_dataset_images)

print(predictions[1])

print(np.argmax(predictions[1]))

print(np.argmax(predictions))
plt.figure()

plt.imshow(fashion_mnist_test_dataset_images[2])

plt.colorbar()

plt.grid(False)

plt.show()
print(fashion_mnist_test_dataset_images.shape)

print(predictions.shape)



Submission = np.empty([0, 2])



for i in range(len(predictions)):

     Submission = np.vstack((Submission,[i, np.argmax(predictions[i])]))

    

Submission = Submission.astype(np.int64)



print(Submission.shape)



print(Submission)

    
data_to_submit = pd.DataFrame(Submission, columns = ["ImageId", "Label"])

data_to_submit.to_csv("submission.csv")

print(data_to_submit)