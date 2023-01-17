# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from math import sqrt, ceil



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Loading training data

data_train = pd.read_csv("../input/train.csv")



# Showing some data

data_train.head()

# Getting first column with labels

y_train = data_train['label']



# Dropping column with labels

x_train = data_train.drop(labels = ["label"], axis = 1)



# Showing current shape of training data

print('Shape of whole data for training', data_train.shape)  # (42000, 785)

print('x_train:', x_train.shape)  # (42000, 784)

print('y_train:', y_train.shape)  # (42000,)



# Showing some examples

%matplotlib inline



# Preparing function for ploting set of examples

# As input it will take 4D tensor and convert it to the grid

# Values will be scaled to the range [0, 255]

def convert_to_grid(x_input):

    N, H, W = x_input.shape

    grid_size = int(ceil(sqrt(N)))

    grid_height = H * grid_size + 1 * (grid_size - 1)

    grid_width = W * grid_size + 1 * (grid_size - 1)

    grid = np.zeros((grid_height, grid_width)) + 255

    next_idx = 0

    y0, y1 = 0, H

    for y in range(grid_size):

        x0, x1 = 0, W

        for x in range(grid_size):

            if next_idx < N:

                img = x_input[next_idx]

                low, high = np.min(img), np.max(img)

                grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)

                next_idx += 1

            x0 += W + 1

            x1 += W + 1

        y0 += H + 1

        y1 += H + 1



    return grid





# Visualizing some examples of training data

examples = np.array(x_train.iloc[:81]).reshape(81, 28, 28)

print(examples.shape)  # (81, 28, 28)



# Plotting

fig = plt.figure()

grid = convert_to_grid(examples)

plt.imshow(grid.astype('uint8'), cmap='gray')

plt.axis('off')

plt.gcf().set_size_inches(7, 7)

plt.title('Some examples of training data', fontsize=24)

plt.show()

plt.close()



# Saving plot

fig.savefig('training_examples.png')

plt.close()

# Loading testing data

x_test = pd.read_csv("../input/test.csv")



# Showing some data

x_test.head()

# Visualizing some examples of testing data

examples = np.array(x_test.iloc[:81]).reshape(81, 28, 28)

print(examples.shape)  # (81, 28, 28)



# Plotting

fig = plt.figure()

grid = convert_to_grid(examples)

plt.imshow(grid.astype('uint8'), cmap='gray')

plt.axis('off')

plt.gcf().set_size_inches(7, 7)

plt.title('Some examples of testing data', fontsize=24)

plt.show()

plt.close()



# Saving plot

fig.savefig('testing_examples.png')

plt.close()

# Making data as numpy array

# Reshaping training and testing data

x_train = np.array(x_train).reshape(-1, 28, 28, 1)

x_test = np.array(x_test).reshape(-1, 28, 28, 1)



# Showing current shape of training and testing data

print('x_train:', x_train.shape)  # (42000, 28, 28, 1)

print('x_test:', x_test.shape)  # (28000, 28, 28, 1)

# Preparing datasets for further using



# Preparing function for preprocessing MNIST datasets for further use in classifier

def pre_process_mnist(x_train, y_train, x_test):

    # Normalizing whole data by dividing /255.0

    x_train = x_train / 255.0

    x_test = x_test / 255.0  # Data for testing consists of 28000 examples from testing dataset



    # Preparing data for training, validation and testing

    # Data for validation is taken with 1000 examples from training dataset in range from 41000 to 42000

    batch_mask = list(range(41000, 42000))

    x_validation = x_train[batch_mask]  # (1000, 28, 28, 1)

    y_validation = y_train[batch_mask]  # (1000,)

    # Data for training is taken with first 41000 examples from training dataset

    batch_mask = list(range(41000))

    x_train = x_train[batch_mask]  # (41000, 28, 28, 1)

    y_train = y_train[batch_mask]  # (41000,)



    # Normalizing data by subtracting mean image and dividing by standard deviation

    # Subtracting the dataset by mean image serves to center the data

    # It helps for each feature to have a similar range and gradients don't go out of control

    # Calculating mean image from training dataset along the rows by specifying 'axis=0'

    mean_image = np.mean(x_train, axis=0)  # numpy.ndarray (28, 28, 1)



    # Calculating standard deviation from training dataset along the rows by specifying 'axis=0'

    std = np.std(x_train, axis=0)  # numpy.ndarray (28, 28, 1)

    # Taking into account that a lot of values are 0, that is why we need to replace it to 1

    # In order to avoid dividing by 0

    for j in range(28):

        for i in range(28):

            if std[i, j, 0] == 0:

                std[i, j, 0] = 1.0



    # Subtracting calculated mean image from pre-processed datasets

    x_train -= mean_image

    x_validation -= mean_image

    x_test -= mean_image



    # Dividing then every dataset by standard deviation

    x_train /= std

    x_validation /= std

    x_test /= std

    

    # Preparing y_train and y_validation for using in Keras

    y_train = to_categorical(y_train, num_classes=10)

    y_validation = to_categorical(y_validation, num_classes=10)



    # Returning result as dictionary

    d_processed = {'x_train': x_train, 'y_train': y_train,

                   'x_validation': x_validation, 'y_validation': y_validation,

                   'x_test': x_test}



    # Returning dictionary

    return d_processed





# Preprocessing data

data = pre_process_mnist(x_train, y_train, x_test)

for i, j in data.items():

    print(i + ':', j.shape)



# x_train: (41000, 28, 28, 1)

# y_train: (41000, 10)

# x_validation: (1000, 28, 28, 1)

# y_validation: (1000, 10)

# x_test: (28000, 28, 28, 1)

model = Sequential()



model.add(Conv2D(64, kernel_size=7, padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=9, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=7, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=7, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))

epochs = 50



h = model.fit(data['x_train'], data['y_train'], batch_size=100, epochs = epochs,

              validation_data = (data['x_validation'], data['y_validation']), callbacks=[annealer], verbose=1)

print("Epochs={0:d}, Train accuracy={1:.5f}, Validation accuracy={2:.5f}".format(epochs, max(h.history['acc']), 

                                                                                 max(h.history['val_acc'])))

%matplotlib inline

plt.rcParams['figure.figsize'] = (15.0, 5.0) # Setting default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



fig = plt.figure()

plt.plot(h.history['acc'], '-o')

plt.plot(h.history['val_acc'], '-o')

plt.title('Model accuracy')

plt.legend(['train', 'validation'], loc='upper left')

plt.xlabel('Epoch', fontsize=15)

plt.ylabel('Accuracy', fontsize=15)

plt.ylim(0.98, 1)

plt.show()



# Saving plot

fig.savefig('model_accuracy.png')

plt.close()

model.save('my_model.h5')
# # Saving model locally without commiting

# from IPython.display import FileLink



# FileLink('my_model.h5')

results = model.predict(data['x_test'])

results = np.argmax(results, axis=1)



# Loading sample template for submission and writing predicted labels into 'Label' column

submission = pd.read_csv('../input/sample_submission.csv')



submission['Label'] = results

submission.to_csv('sample_submission.csv', index=None)

# Cheking

s = pd.read_csv('sample_submission.csv')

s.head()

# # Saving resulted data locally without commiting

# from IPython.display import FileLink



# FileLink('sample_submission.csv')
