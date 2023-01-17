import tensorflow as tf                       # deep learning library

import numpy as np                            # for matrix operations

import matplotlib.pyplot as plt               # for visualization

%matplotlib inline
from tensorflow.keras.datasets.mnist import load_data    # To load the MNIST digit dataset



(X_train, y_train) , (X_test, y_test) = load_data()      # Loading data
print("There are ", len(X_train), "images in the training dataset")     # checking total number of records / data points available in the X_train dataset

print("There are ", len(X_test), "images in the test dataset")     # checking total number of records / data points available in the X_test dataset
# Checking the shape of one image

X_train[0].shape
# Take a look how one image looks like

X_train[0]
plt.matshow(X_train[0])
# we can use y_train to cross check

y_train[0]
# code to view the images

num_rows, num_cols = 2, 5

f, ax = plt.subplots(num_rows, num_cols, figsize=(12,5),

                     gridspec_kw={'wspace':0.03, 'hspace':0.01}, 

                     squeeze=True)



for r in range(num_rows):

    for c in range(num_cols):

      

        image_index = r * 5 + c

        ax[r,c].axis("off")

        ax[r,c].imshow( X_train[image_index], cmap='gray')

        ax[r,c].set_title('No. %d' % y_train[image_index])

plt.show()

plt.close()


X_train = X_train / 255

X_test = X_test / 255



"""

Why divided by 255?

The pixel value lie in the range 0 - 255 representing the RGB (Red Green Blue) value. """
X_train[0]
X_train.shape
X_train_flattened = X_train.reshape(len(X_train), 28*28)    # converting our 2D array representin an image to one dimensional

X_test_flattened = X_test.reshape(len(X_test), 28*28)
X_train_flattened.shape
# Defining the Model

model = tf.keras.Sequential([

    tf.keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')     # The input shape is 784. 

])
model.summary()
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(X_train_flattened, y_train, epochs=5)
model.evaluate(X_test_flattened, y_test)
y_predicted = model.predict(X_test_flattened)

y_predicted[0]
np.argmax(y_predicted[0])
plt.matshow(X_test[0])
# Defining the model

model = tf.keras.Sequential([

    tf.keras.layers.Dense(100, input_shape=(784,), activation='relu'),

    tf.keras.layers.Dense(10, activation='sigmoid')

])

model.summary()

# Compiling the model

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



# Fit the model

model.fit(X_train_flattened, y_train, batch_size= 128,epochs=20)
# Evaluate the model

model.evaluate(X_test_flattened,y_test)
model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(100, activation='relu'),

    tf.keras.layers.Dense(10, activation='sigmoid')

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=10)