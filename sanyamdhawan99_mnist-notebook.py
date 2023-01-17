import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

from keras.optimizers import Adam, RMSprop

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator



import warnings

warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')

train_data.shape, test_data.shape
train_labels = train_data['label']

del train_data['label']

train_labels.shape, train_data.shape
# sns.countplot(train_labels)

label, count = np.unique(train_labels, return_counts = True)

print(train_labels.value_counts())

print('-'*50)

print('There are', train_labels.isnull().sum() ,'null values in labels')

print('-'*50)

sns.barplot(label, count)
# converting the data to a grayscale and 28*28 image

train_data = train_data / 255.0

test_data = test_data / 255.0

x_train = train_data.values.reshape(-1, 28, 28, 1)

x_test = test_data.values.reshape(-1, 28, 28,1)
# one hot encoding labels

y_train = to_categorical(train_labels, 10)
# split data to training and validation sets (validation = 10% of the data)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 1)

x_train.shape, y_train.shape, x_val.shape, y_val.shape
fig = plt.figure(figsize = (8, 8))

row = 3

col = 4

for i in range(row*col):

    fig.add_subplot(row, col, i+1)

    plt.imshow(x_train[i][:, :, 0])

plt.show()
model = Sequential()

# set 1

model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'SAME', activation = 'relu', input_shape = (28, 28, 1)))

model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'SAME', activation = 'relu'))

model.add(MaxPool2D(2, 2))

model.add(Dropout(0.25))



# set 2

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'SAME', activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'SAME', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = 'relu')) # hidden later

model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax')) # output layer
# optimizer = Adam(learning_rate=0.001)

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# compile the model

model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
# setting up an annealer

LR_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=0.00001)
# define epochs and batch size

epochs = 30

batch_size = 86
image_data = ImageDataGenerator(

                featurewise_center = False,

                samplewise_center = False,

                featurewise_std_normalization = False,

                samplewise_std_normalization = False,

                zca_whitening = False,

                rotation_range = 10,

                width_shift_range = 0.1,

                height_shift_range = 0.1,

                zoom_range = 0.1,

                horizontal_flip = False,

                vertical_flip= False

            )
# fit the training data on ImageData Generator

image_data.fit(x_train)
# fit the model

train_generator = image_data.flow(x_train, y_train, batch_size = batch_size)

trained_model = model.fit_generator(train_generator, epochs=epochs, validation_data=(x_val, y_val), verbose=1, steps_per_epoch=len(x_train)//batch_size,

                   callbacks=[LR_reducer])
# printing training and validation loss and accuracy

fig, ax = plt.subplots(2, 1)

ax[0].plot(trained_model.history['loss'], color = 'b', label = 'Training Loss')

ax[0].plot(trained_model.history['val_loss'], color = 'r', label = 'Validation Loss')

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(trained_model.history['accuracy'], color = 'b', label = 'Training Accuracy')

ax[1].plot(trained_model.history['val_accuracy'], color = 'r', label = 'Validation Accuracy')

legend = ax[1].legend(loc='best', shadow=True)
# printing confusion matrix

val_pred = model.predict(x_val)

val_pred_classes = np.argmax(val_pred, axis = 1)

val_true_classes = np.argmax(y_val, axis = 1)

results = confusion_matrix(val_true_classes, val_pred_classes)

print(results)
# printing some errorneous data

error_data = (val_pred_classes - val_true_classes != 0)

error_pred = val_pred[error_data]

error_pred_classes = val_pred_classes[error_data]

error_true = y_val[error_data]

error_true_classes = val_true_classes[error_data]

error_x_val = x_val[error_data]

print('Number of wrong predcitions in validation data = ', error_data.sum())

fig = plt.figure(figsize = (8, 8))

row = 3

col = 4

for i in range(row*col):

    fig.add_subplot(row, col, i+1)

    plt.imshow(error_x_val[i][:, :, 0])

    plt.title('Predicted Label: ' + str(error_pred_classes[i]) + '\nTrue Label: ' + str(error_true_classes[i]))

plt.show()
results = model.predict(x_test)

results_classes = np.argmax(results, axis = 1)
result_data = {'ImageID': [i for i in range(1, len(results_classes)+1)],

              'Label': results_classes

              }

results_df = pd.DataFrame(result_data)

results_df.head()
results_df.to_csv('MNIST_data_output_using_CNN_30_epochs.csv', index = False)