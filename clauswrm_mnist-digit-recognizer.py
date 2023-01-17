# Imports
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.datasets import mnist

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

%matplotlib inline
from matplotlib import pyplot as plt

import numpy as np
# Set the seed for consistent results
np.random.seed(1337)
# Collect data
train = pd.read_csv("../input/train.csv")
x_test = pd.read_csv("../input/test.csv")

y_train_entire = train["label"]
x_train_entire = train.drop(labels = ["label"],axis = 1)

# ...for good measure
shuffle(x_train_entire, y_train_entire)

# Split in train/validation
x_train, x_validate, y_train, y_validate = train_test_split(x_train_entire, y_train_entire,
                                                            test_size = 0.1)

print('Train    = ', len(x_train))
print('Validate = ', len(x_validate))
print('Test     = ', len(x_test))

# Reshape to a valid tensor
x_train_conv = x_train.values.reshape(x_train.shape[0], 28, 28, 1)
x_validate_conv = x_validate.values.reshape(x_validate.shape[0], 28, 28, 1)
x_test_conv = x_test.values.reshape(x_test.shape[0], 28, 28, 1)

# Casts the arrays to valid number type
x_train_conv = x_train_conv.astype('float32')
x_validate_conv = x_validate_conv.astype('float32')
x_test_conv = x_test_conv.astype('float32')

#One-hot encode the labels
y_train_one_hot = to_categorical(y_train, 10)
y_validate_one_hot = to_categorical(y_validate, 10)
# Adapt the data

# Normalize the features from [0-255] to [0-1] 
x_train_conv /= 255
x_validate_conv /= 255
x_test_conv /= 255
# Build model
model_name = '3conv_128-256_2max'
model = Sequential()

model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Compile model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

model.summary()
# Train model
batch_size = 128

history = model.fit(x_train_conv, y_train_one_hot,
                    batch_size=batch_size,
                    epochs=10,
                    verbose=2,
                    validation_data=(x_validate_conv, y_validate_one_hot))

#model.save_weights('persistence/' + model_name + '_weights.h5')
#model.save('persistence/' + model_name + '_keras.h5')
# Visualize training

def plot_training_history(history):
    """ Plot helper function """
    print('Availible variables to plot: {}'.format(history.history.keys()))
    line_style = '.-'
    
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'], line_style)
    plt.plot(history.history['val_acc'], line_style)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Model Accuracy over Epochs')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], line_style)
    plt.plot(history.history['val_loss'], line_style)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Model Loss over Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid()
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    plt.show()


plot_training_history(history)
# Predict test set
predictions = model.predict(x_test_conv, verbose=0)

# Select index with max probability '[0.1, 0.01, 0, 0, 1, 0, 0.1]' -> '4'
predicted_nums = np.argmax(predictions, axis = 1)
# Make result submission csv for Kaggle (https://www.kaggle.com/c/digit-recognizer)
results = pd.Series(predicted_nums, name='Label')

submission = pd.concat((pd.Series(range(1,28001), name='ImageId'), results), axis = 1)
submission.to_csv(model_name + '_submission.csv', index=False)