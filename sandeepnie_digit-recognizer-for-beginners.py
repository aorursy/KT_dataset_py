# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





#Tensorflow and tf.keras

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_images = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sample = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
#extract the label value from the training dataset, not required from test data as labels are not given for this test dataset

train_label = np.array(train['label'])

train_images = train.drop('label', axis=1)
type(train_label)
onehotencoder = OneHotEncoder() 

  

binarised_train_label = onehotencoder.fit_transform(train_label.reshape(-1,1)).toarray() 
#check one samle data if one hot encoding is working fine.

print(binarised_train_label[5])

print(train_label[5])
train_images.shape
test_images.shape
print("Datatype for train_images = ",type(train_images))

print("Datatype for test_images = ",type(test_images))
train_images.head()
len(test_images)
#training dataset is in 784 pixels which needs to be reshaped usign the re-shape function

train_images_arr = np.array(train_images).reshape(42000,28,28)

test_images_arr = np.array(test_images).reshape(28000,28,28)
print("Datatype for train_images = ",type(train_images))

print("Datatype for test_images = ",type(test_images))
#Lets see one of the examples from the image dataset

plt.figure()

plt.imshow(train_images_arr[13])

plt.colorbar()

plt.grid(False)

plt.show()
train_images = train_images / 255



test_images = test_images / 255
train_images = train_images.values.reshape(-1,28,28,1)

test_images = test_images.values.reshape(-1,28,28,1)
plt.figure(figsize =(10,10))

for i in range(25):

    plt.subplot(5,5,1+i)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(train_images_arr[i])

    plt.xlabel(train_label[i])
train_images.shape
test_images.shape
#lets reshape the data into 3 dimensions

train_images = train_images.reshape(len(train_images),28,28,1)

test_images = test_images.reshape(len(test_images),28,28,1)
X_train_images, X_test_images, Y_train_labels, Y_test_labels = train_test_split(train_images, train_label,test_size=0.20, random_state=42)
X_train_images.shape
#model = keras.Sequential([

#    keras.layers.Flatten(input_shape =(28,28)),

#    keras.layers.Dense(128, activation=tf.nn.relu),

#    keras.layers.Dense(10, activation=tf.nn.softmax)

#])
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

              loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy'])

model.summary()
X_train_images.shape
model.fit(X_train_images,Y_train_labels,epochs=10, validation_data=(X_test_images, Y_test_labels))
test_loss, test_acc = model.evaluate(X_test_images,Y_test_labels)



print("Test accuracy : ", test_acc)
predictions = model.predict(X_test_images)
predictions[0]
predictions = model.predict_classes(test_images, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)
submissions.head()