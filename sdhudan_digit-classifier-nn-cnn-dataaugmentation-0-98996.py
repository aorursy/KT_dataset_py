import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# TensorFlow / Keras functions
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.datasets import mnist

import warnings
warnings.filterwarnings('ignore')
!pwd
# Read in the data
train_raw = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_raw.shape, test.shape
train_raw.head()
# Split raw data into predictor variables 'X' and response variable 'y'
X = train_raw.drop('label', axis = 1).values/255  # Normalize the data
y = train_raw.label.values
X.shape, y.shape
# Check the distribution of the digits in training set
sns.countplot(y)
# Display the first 8 images in the training set
fig, ax = plt.subplots(2, 4) 
m=0
while m < 8:
    for i in range(2):
        for j in range(4):
            ax[i, j].imshow(X[m].reshape(28, 28), plt.cm.binary) 
            ax[i, j].set_xticks(())
            ax[i, j].set_yticks(())
            m += 1
fig.set_size_inches(16, 7) 
fig.tight_layout()
# Check the first 8 digits in 'y'
y[:8]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 18000, stratify = y, random_state = 2020)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size = 2000, stratify = y_valid, 
                                                    random_state = 2020)

X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape
# Reshape the input data into 3 dimension 
X_train = X_train.reshape(24000, 28, 28, 1)
X_valid = X_valid.reshape(16000, 28, 28, 1)
X_test = X_test.reshape(2000, 28, 28, 1)

X_train.shape, X_valid.shape, X_test.shape
# Label encode y variables
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)
y_train[:8]
datagen = ImageDataGenerator(
        rotation_range=8,  # random rotations within degree range: 0 - 15, because of hand writing issues
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally by 10% of total width
        height_shift_range=0.1)  # randomly shift images vertically by 10% of total height

datagen.fit(X_train)
nn = Sequential()
nn.add(Flatten(input_shape=(28, 28, 1)))
nn.add(Dense(16, activation='relu')) 
nn.add(Dense(32, activation='relu'))
nn.add(Dense(64, activation='relu'))
nn.add(Dense(10, activation='softmax'))
nn.summary()
# Compile
nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit
nn.fit(datagen.flow(X_train, y_train, batch_size=128), epochs = 30, verbose = 2, validation_data = (X_valid, y_valid))
nn.evaluate(X_test, y_test)
# Make predictions using NN
test = test.values / 255
test_reshape = test.reshape(test.shape[0], 28, 28, 1)
nn_results = nn.predict_classes(test_reshape)
nn_results
# nn_submission = pd.DataFrame(np.concatenate((np.array(range(1, test.shape[0] + 1)).reshape(-1, 1), 
#                                              nn_results.reshape(-1, 1)), axis = 1), 
#                              columns=['ImageId', 'Label'])

# nn_submission.head()
# nn_submission.to_csv('submission_NN.csv', header=True, index=False)

cnn = Sequential()
cnn.add(Conv2D(8, kernel_size=5, activation='relu', padding='same', input_shape=(28,28,1))) 
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2))
cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2)) 
cnn.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
cnn.add(Dropout(0.5))

cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))
cnn.summary()

# Compile the model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = cnn.fit(datagen.flow(X_train, y_train, batch_size=128), epochs = 50, verbose = 2, validation_data = (X_valid, y_valid))
cnn.evaluate(X_test, y_test)
# Make predictions
cnn_results = cnn.predict_classes(test_reshape)
cnn_results
# Plot the accuracy score in both training set and validation for the CNN
plt.plot(history.history['accuracy'], label='Training loss') 
plt.plot(history.history['val_accuracy'], label='Validation loss') 
plt.legend()
# Plot the loss in both training set and validation for the CNN
plt.plot(history.history['loss'], label='Training loss') 
plt.plot(history.history['val_loss'], label='Validation loss') 
plt.legend()
cnn_submission = pd.DataFrame(np.concatenate((np.array(range(1, test.shape[0] + 1)).reshape(-1, 1), 
                                             cnn_results.reshape(-1, 1)), axis = 1), 
                             columns=['ImageId', 'Label'])

cnn_submission.to_csv('cnn_submission.csv', header=True, index=False)
