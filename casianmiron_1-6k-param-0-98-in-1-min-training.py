import tensorflow as tf

import pandas as pd

from tensorflow.keras import  utils



# Load the data

train = pd.read_csv("../input/train.csv")

Y_train = train["label"]

training_images = train.drop(labels = ["label"],axis = 1) 

training_images=training_images.values.reshape(-1, 28, 28, 1)

training_images=training_images / 255.0

# Create model

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(8, (7,7),strides=(1, 1), activation='relu', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(5, 5),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(10, activation='softmax')

])



#Optimizer + summary of the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
#Train the model

model.fit(training_images, Y_train, epochs=30)

import numpy as np

test = pd.read_csv("../input/test.csv")

test = test / 255.0

test=test.values.reshape(-1,28,28,1)

results = model.predict(test)

# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)
model.save('mmymnist.h5')
