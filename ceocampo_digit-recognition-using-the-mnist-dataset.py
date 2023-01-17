# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style()

%matplotlib inline
# Importing MNIST dataset

df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')
y_train = df_train['label']

y_train
x_train = df_train.drop('label', axis=1)

x_train = x_train.values

x_train
x_train.shape
# Reshaping array to visualize the images

plt.imshow(x_train[0].reshape(28,28), cmap='Greys')

plt.show()
x_train = x_train.reshape(42000,28,28)
x_train.shape
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=19)
x_train.shape
x_valid.shape
from tensorflow.keras.utils import to_categorical

y_cat_valid = to_categorical(y_valid, num_classes=10)

y_cat_train = to_categorical(y_train, num_classes=10)
# Sampling first image visually

first_image = x_train[0]

plt.imshow(first_image, cmap='Greys')

plt.show()
print('Actual image:')

print(y_train[0])
first_image.max()
first_image.min()
# Scaling values on train and test data

x_train_scaled = x_train/255

x_valid_scaled = x_valid/255
# Rechecking image after scaling

plt.imshow(x_train_scaled[0], cmap='Greys')

plt.show()
# Reshaping final training and testing data to prep for training

# (batch_size, width, height, color_channels)

x_train_final = x_train_scaled.reshape(37800,28,28,1)

x_valid_final = x_valid_scaled.reshape(4200,28,28,1)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
# Building CNN Model



# Instantiate model

model = Sequential()



# Convolution layer 1

model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))

# Convolution layer 2

model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))

# Pooling layer (selected half of kernel_size)

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Convolution layer 3

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))

# Convolution layer 4

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))

# Pooling layer (selected half of kernel_size)

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Flattening image

model.add(Flatten())

# Dense layer

model.add(Dense(256, activation='relu'))



# Output layer

model.add(Dense(10, activation='softmax'))



# Compiling model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=10,

                              width_shift_range=0.1,

                              height_shift_range=0.1,

                              shear_range=0.1,

                              zoom_range=0.1,

                              fill_mode='nearest')
train_image_gen = image_gen.fit(x_train_final)
train_image_gen
model.fit_generator(image_gen.flow(x_train_final, y_cat_train), epochs=10, validation_data=(x_valid_final, y_cat_valid), callbacks=[early_stop])
metrics = pd.DataFrame(model.history.history)

metrics
metrics[['loss', 'val_loss']].plot()

plt.show()
metrics[['accuracy', 'val_accuracy']].plot()

plt.show()
model.evaluate(x_valid_final, y_cat_valid, verbose=0)
y_pred = model.predict_classes(x_valid_final)

y_pred
# Model evaluation

from sklearn.metrics import classification_report, confusion_matrix

print('Classification Report:')

print(classification_report(y_valid, y_pred))

print('\n')

print('Confusion Matrix:')

print(confusion_matrix(y_valid, y_pred))
np.random.seed(19)

random_selection = np.random.randint(0, 4201, size=1)

random_sample = x_valid_final[random_selection]

plt.imshow(random_sample.reshape(28,28), cmap='Greys')

plt.show()
print('Prediction:')

print(model.predict_classes(random_sample.reshape(1,28,28,1))[0])
np.random.seed(20)

random_selection_2 = np.random.randint(0, 4201, size=1)

random_sample_2 = x_valid_final[random_selection_2]

plt.imshow(random_sample_2.reshape(28,28), cmap='Greys')

plt.show()
print('Prediction:')

print(model.predict_classes(random_sample_2.reshape(1,28,28,1))[0])
np.random.seed(22)

random_selection_3 = np.random.randint(0, 4201, size=1)

random_sample_3 = x_valid_final[random_selection_3]

plt.imshow(random_sample_3.reshape(28,28), cmap='Greys')

plt.show()
print('Prediction:')

print(model.predict_classes(random_sample_3.reshape(1,28,28,1))[0])
# Reshaping test data

x_test = df_test.values

x_test = x_test.reshape(28000,28,28)

x_test.shape
# Scaling test data

x_test_scaled = x_test/255
# Generating preditions

test_predictions = model.predict_classes(x_test_scaled.reshape(28000,28,28,1))
test_predictions
# Save test predictions to file

output = pd.DataFrame({'ImageId': df_test.index + 1,

                       'Label': test_predictions})

output.to_csv('submission.csv', index=False)