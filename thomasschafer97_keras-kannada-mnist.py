import numpy as np

import pandas as pd

import os, keras, math

from keras.utils import to_categorical 

from keras.preprocessing.image import ImageDataGenerator

from keras import layers, models, regularizers



# Putting the data in to pandas DataFrames

train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

extra_train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
# Formatting the data

def data_prep(dataframe):

    array = dataframe.values

    array = array[:, 1:]

    array = array.reshape(array.shape[0], 28, 28, 1)

    array = array.astype('float32')/255

    return array





train_X = data_prep(train_data)

test_X = data_prep(test_data)



train_y = to_categorical((train_data.values)[:, 0])



print(train_X.shape, train_y.shape, test_X.shape)
# We will be using data augmentation to allow us to simulate having a larger dataset

datagen = ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2)



datagen.fit(train_X)
# Defining a function to build the model

def build_model():

    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))

    model.add(layers.BatchNormalization(axis=1))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.BatchNormalization(axis=1))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model





# Building and fitting a model with a 0.1 validation split

train_X_short = train_X[:54000]

train_y_short = train_y[:54000]

val_X = train_X[54000:]

val_y = train_y[54000:]

print(train_X_short.shape, train_y_short.shape, val_X.shape, val_y.shape)



model = build_model()

history = model.fit_generator(datagen.flow(train_X_short, train_y_short, batch_size=32),

                              validation_data=(val_X, val_y), steps_per_epoch=100, epochs=300, verbose=0)
# Plotting the training and validation accuracy

import matplotlib.pyplot as plt

%matplotlib inline



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

epochs = [i+1 for i in range(len(acc))]



acc_short = []

val_acc_short = []

epochs_short = []



for i in range(len(epochs)):

    if i % (len(epochs)//30) == 0:

        epochs_short.append(i)

        acc_short.append(acc[i])

        val_acc_short.append(val_acc[i])

        



plt.plot(epochs_short, acc_short, 'o', label='Training acc')

plt.plot(epochs_short, val_acc_short, 'b', label='Validation acc')

plt.title('Accuracy')

plt.legend()

plt.show()
# Deciding how many epochs to train the model with based on the validation accuracy

# To smooth out any random variation we will find which triplet of epochs gives the lowest average accuracy

#   and choose the point in the middle



if len(epochs) >= 4:

    # Defining a dictionary that will hold the averages of triplets of consecutive epochs

    triple_averages = {}

    for i in range(1, len(epochs)-1):

        triple_averages[i] = (val_acc[i-1] + val_acc[i] + val_acc[i+1])/3

    # Finding the triplet giving the highest average, and selecting the point in the middle

    for i in range(1, len(triple_averages)+1):

        min_avg = max(list(triple_averages.values()))

        if triple_averages[i] == min_avg:

            epochs_num = i

else:

    for i in range(len(epochs)):

        if val_acc[i] == max(val_acc):

            epochs_num = i+1

            

epochs_num *= 10/9
# Building a fresh model with no validation data

model = build_model()

history = model.fit(train_X, train_y, batch_size=32, epochs=int(epochs_num), verbose=0)
# Outputting the predictions

predictions = model.predict(test_X)

predictions_new = np.argmax(predictions, axis=1)

output = pd.DataFrame({'id': test_data.id, 'label': predictions_new})

output.to_csv('submission.csv', index=False)

print('Complete')