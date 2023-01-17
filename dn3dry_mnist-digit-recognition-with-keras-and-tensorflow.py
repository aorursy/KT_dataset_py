# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#load the libs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
print(tf.__version__)
print(os.listdir("../input"))
#import data and define the classes
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
class_names = [0,1,2,3,4,5,6,7,8,9]

#print out training data
print(train_data.shape)
print(train_data.head())
#split out the data into features (pixel values) and categorical labels (digit values 0-9)
train_x = train_data.iloc[:,1:].values.astype('float32') # all pixel values
train_y = train_data.iloc[:,0].values.astype('int32') # only labels i.e targets digits

test_x = test_data.iloc[:,].values.astype('float32') # all pixel values

#reshape the features to be 28x28
train_x = train_x.reshape(train_x.shape[:1] + (28, 28, 1))
test_x = test_x.reshape(test_x.shape[:1] + (28, 28, 1))

#change the labels to be one-hot encoded
train_y = keras.utils.to_categorical(train_y)
num_classes = train_y.shape[1]

#normalize pixel values using minmax (values between 0 and 1 inclusive)
train_x = train_x / 255
test_x = test_x / 255
# Create callbacks for tensorboard and early stopping
my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, min_delta=0, monitor='val_loss')]
plt.figure()
plt.imshow(train_x[0].reshape(28, 28))
plt.colorbar()
plt.grid(False)
plt.show()

#plot a group of features and labels to check data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_x[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(train_y[i])])
plt.show()
#define the model and layers
model = keras.Sequential([ 
    #start: 1@28x28 image matrices

    #convolution with 32 filters that use a 3x3 kernel (convolution window) and stride of 1
    keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1), strides=(1,1), activation='relu'),
    #now: 32@26x26

    #subsampling using max pooling and a 2x2 filter (largest element from the rectified feature map)
    keras.layers.MaxPool2D(pool_size=(2,2)),
    #now: 32@13x13 matricies

    #convolution with 64 filters that use a 3x3 kernel (convolution window) and stride of 1
    keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu'),
    #now: 64@11x11

    #subsampling using max pooling
    keras.layers.MaxPool2D(pool_size=(2,2)),
    #now: 64@5x5

    #flatten to a single vector
    keras.layers.Flatten(),
    #now: flattened to 1600

    #first fully connected layer with 128 units
    keras.layers.Dense(128, activation=tf.nn.relu),

    #drop 20% of units to help prevent overfitting
    keras.layers.Dropout(0.2),

    #softmax layer for classification
    keras.layers.Dense(num_classes, activation=tf.nn.softmax)
])
#compile the model
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#print a summary of the model
model.summary()
#train the model
model.fit(x=train_x, 
            y=train_y,
            batch_size=32,
            epochs=30,
            verbose=1,
            callbacks = my_callbacks,
            validation_split=0.05,
            shuffle=True
            )

#make predictions on the test features
predictions = model.predict(test_x)
def plot_value_array(i, predictions_array):
    predictions_array = predictions_array[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')

def plot_image(i, predictions_array, img):
    img = img.reshape(img.shape[0] ,28, 28)
    predictions_array, img = predictions_array[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    plt.xlabel("{} - prob:{:2.0f}%".format(class_names[predicted_label], 100*np.max(predictions_array)), color='red')

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_x)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions)
plt.show()
#submissions for Kaggle
#cat_predictions = np.argmax(predictions, axis=1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": np.argmax(predictions, axis=1)})
submissions.to_csv("my_submissions.csv", index=False, header=True)