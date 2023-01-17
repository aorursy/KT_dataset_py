#importing libraries..

import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

import keras

from keras.utils import np_utils

import tensorflow as tf

import seaborn as sns
np.random.seed(9)

tf.random.set_seed(9)
#loading train_data

train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv") #(42000, 785)

train_data.head()
#slicing train_data into train_labels and train_images

train_labels = train_data.iloc[:,0].astype(np.float32).values   #(42000, 1) <class 'numpy.ndarray'>

train_images = train_data.iloc[:,1:].astype(np.float32).values   #(42000, 784) <class 'numpy.ndarray'>



#reshaping train_images

train_images = train_images.reshape(42000, 28, 28, 1)   #(42000, 28, 28, 1)  
#splitting train_data into train and dev sets

x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.024, random_state = 42, shuffle=True, stratify=train_labels)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape )
train_set_plot = sns.countplot(y_train)
dev_set_plot = sns.countplot(y_test)
#one-hot enoding 

y_train = np_utils.to_categorical(y_train, 10)  

y_test = np_utils.to_categorical(y_test, 10)   



#normalizing train and dev set images

x_train = x_train / 255.0  

x_test = x_test / 255.0
#loading test_data

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")  #(28000, 784)

test_data.head()
test_images = test_data.astype(np.float32).values # (28000, 784) <class 'numpy.ndarray'>



#reshaping test_images

test_images = test_images.reshape(28000, 28, 28, 1) #(28000, 28, 28, 1)



#normalizing test_images

test_images = test_images / 255.0
#Model



model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(32, (5,5), activation='relu', padding = 'Same', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding = 'Same'),

  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding = 'Same'),

  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding = 'Same'),

  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Dropout(0.25),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()
#training model using train set

model.fit(x_train, y_train, epochs=30)
#evaluating model using dev set

test_loss = model.evaluate(x_test, y_test)
#making predictions using test_images

predictions = model.predict(test_images)

predictions = np.argmax(predictions, axis=1)
#making submission file



predictions = pd.Series(predictions,name="Label")

result = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)



result.to_csv("digit_recognizer.csv",index=False)