import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta # I believe this is better optimizer for our case
from keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from sklearn.model_selection import train_test_split # to split our train data into train and validation sets
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(13) # My lucky number
num_classes = 10 # We have 10 digits to identify
batch_size = 128 # Handle 128 pictures at each round
epochs = 10 # 10 Epoch is enough for %99.4 Accuracy!!!!
img_rows, img_cols = 28, 28 # Image dimensions 28 pixels in height&width
input_shape = (img_rows, img_cols,1) # We'll use this while building layers
# Load some date to rock'n roll
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Drop the label from the data and move it to real label part
y_train = train["label"]
x_train = train.drop(labels = ["label"],axis = 1 )
# Normalize both sets
x_train /= 255
test /= 255
print(x_train.shape[0], 'train samples')
print(test.shape[0], 'test samples')
# Images should be in shape of height,width and color channel so it will be 28x28x1
x_train = x_train.values.reshape(-1,img_rows,img_cols,1)
test = test.values.reshape(-1,img_rows,img_cols,1)
# Class vectors needs to be binary so we use "to_catogorical" function of keras utilities for one-hot-encoding
y_train = keras.utils.to_categorical(y_train, num_classes = num_classes)
# Lets split our train set into train and validation test sets with my lucky number 13 :)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=13)
model = Sequential()

# Add convolutional layer consisting of 32 filters and shape of 5x5 with ReLU activation
# We want to preserve more information for followin layers so we start using padding
# 'Same' padding tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = input_shape))
BatchNormalization(axis=-1)
# Add convolutional layer consisting of 32 filters and shape of 5x5 with ReLU activation
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

# Add Maxpool layer with the shape of 2x2
model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)
# Dropping %25 of neurons
model.add(Dropout(0.25))

# Add convolutional layer consisting of 64 filters and shape of 3x3 with ReLU activation
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
BatchNormalization(axis=-1)
# Add convolutional layer consisting of 64 filters and shape of 3x3 with ReLU activation
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
# Add convolutional layer consisting of 64 filters and shape of 3x3 with ReLU activation
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
# Add Maxpool layer with the shape of 2x2 and strides for controlling convolutions over input volume
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# Dropping %25 of neurons
model.add(Dropout(0.25))

# To be able to merge into fully connected layer we have to flatten
model.add(Flatten())
BatchNormalization()
# Adding fully connected layer with 256 ReLU activated neurons
model.add(Dense(256, activation = "relu"))
BatchNormalization()
# Dropping %50 of neurons
model.add(Dropout(0.5))

# Lets add softmax activated neurons as much as number of classes
model.add(Dense(num_classes, activation = "softmax"))
# Adadelta (my favorite) inorder to get over %99 before 5th epoch
optimizer = Adadelta()
# Compile the model with loss and metrics
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Generate batches of tensor image data with real-time data augmentation more detail: https://keras.io/preprocessing/image/
datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train)
# Start model training with the batch size
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size)
# Evaluate accuracy and loss over validation set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])