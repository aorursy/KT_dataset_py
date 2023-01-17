import numpy as np # For maths operations

import mnist # This our dataset's library

from keras.models import Sequential # This our model for generating neural network

from keras.layers import Dense # We can add layers
## Now we prepare the our data



# It will download Mnist dataset, it will take the time

train_images = mnist.train_images()

train_labels = mnist.train_labels()



test_images = mnist.test_images()

test_labels = mnist.test_labels()
print(train_images.shape) 

print(train_labels.shape)
# Normalize the images.

train_images = (train_images / 255) - 0.5

test_images = (test_images / 255) - 0.5



# Flatten the images.

train_images = train_images.reshape((-1, 784))

test_images = test_images.reshape((-1, 784))
print(train_images.shape)

print(test_images.shape)
# Build the model.

model = Sequential([

  Dense(64, activation='relu', input_shape=(784,)),

  Dense(64, activation='relu'),

  Dense(10, activation='softmax'),

])
# Compile the model.

model.compile(

  optimizer='adam',

  loss='categorical_crossentropy',

  metrics=['accuracy'],

)
# Train the model.

from keras.utils import to_categorical

# It turns our array of class integers into an array of one-hot vectors instead. For example, 2 would become [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (it’s zero-indexed)



model.fit(train_images,to_categorical(train_labels),

  epochs=5,

  batch_size=32,

)
# Predict on the first 5 test images.

predictions = model.predict(test_images[:7])
# This is our predictions

print(np.argmax(predictions, axis=1))
# For print a orignal Label

print(test_labels[:7])