# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
%matplotlib inline

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
# Load the data
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
# Split input data and label
Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)

del train
# Check train datasets
print("Training data info: ")
print(X_train.shape)
print(Y_train.shape)
print(Y_train.value_counts(), "\n")

print("Test data info:")
print(test.shape, "\n")

X_train.isnull().any().describe()
# [0,1] Normalization
X_train = X_train / 255
test = test / 255
# Reshape [784,1] data to [28px, 28px, 1]
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# train data label one-hot encoding (2->[0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
# Show a example of data
plt.imshow(X_train[0][:,:,0])
# Define the CNN model
model = Sequential()

# L1: 1st Conv layer (input:28x28 x1, output:24x24 x20)
model.add(Conv2D(input_shape=(28,28,1), kernel_size=(5,5), 
                 filters=20, activation = "relu"))

# L2: 1st Pooling layer (input:24x24 x20, output:12x12 x20)
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

# L3: 2nd Conv layer (input:12x12 x20, output:8x8 x50)
model.add(Conv2D(kernel_size=(5,5), filters=50, activation='relu'))

# L4: 2nd Pooling layer (input:8x8 x50, output:4x4 x50)
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

# L5: Flatten layer(input:4x4 x50, output:)
model.add(Flatten())

# L6: Fully Connected Network(FCN) layer (input: , output:500x1)
model.add(Dense(500, activation='relu'))

# L7: FCN layer (input:500, output:10x1)
model.add(Dense(10, activation='softmax'))
# Check CNN model structure
model.build()
model.summary()
# compile CNN model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])
# Training NN model
history = model.fit(X_train, Y_train, validation_split=0.1, 
                    epochs=10, batch_size=32)
# plot the training rate
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Model accuracy')
plt.axis((0, 10, 0, 1.1))
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train', 'Test'], loc='best')
plt.show()
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),
                        results],axis = 1)

submission.to_csv("cnn_mnist_datagen20201011.csv",index=False)


