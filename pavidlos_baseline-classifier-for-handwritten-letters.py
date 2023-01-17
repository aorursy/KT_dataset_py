import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pylab as plt

from matplotlib import cm



from sklearn.model_selection import train_test_split



from tensorflow.python import keras

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D





from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
# Load h5 file

import h5py

file = h5py.File("../input/LetterColorImages_123.h5", 'r')

# List all groups

keys = list(file.keys())

keys 
# Create tensors and targets

backgrounds = np.array(file[keys[0]])

tensors = np.array(file[keys[1]])

targets = np.array(file[keys[2]])

img_rows = tensors.shape[1]

img_cols = tensors.shape[2]

img_color_channels = tensors.shape[3]

num_classes = np.unique(targets).size

print ('Tensor shape:', tensors.shape)

print ('Target shape', targets.shape)

print ('Background shape:', backgrounds.shape)
# Normalize the tensors

tensors = tensors.astype('float32')/255
# Read and display a tensor using Matplotlib

print('Label: ', targets[50])

plt.figure(figsize=(3,3))

plt.imshow(tensors[50]);

num_classes
# One-hot encoding the targets, started from the zero label

cat_targets = to_categorical(targets-1, num_classes=num_classes)

cat_targets.shape
# Split the data for training and testing

x_train, x_test, y_train, y_test = train_test_split(tensors, 

                                                    cat_targets, 

                                                    test_size = 0.2, 

                                                    random_state = 1990)
# Create model

model = Sequential()



model.add(Conv2D(32, kernel_size=(3,3), 

                 activation='relu', 

                 input_shape=(img_rows, img_cols, img_color_channels)))

model.add(Conv2D(32, kernel_size=(3, 3), 

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=(3, 3), 

                 activation='relu'))

model.add(Conv2D(64, kernel_size=(3, 3), 

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))



# model.add(Conv2D(128, kernel_size=(3, 3), 

#                  activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.5))



model.add(Flatten())

# model.add(Dropout(0.5))

# model.add(Dense(256, activation='relu'))

# model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
# Train

history = model.fit(x_train, y_train,

                    batch_size=256,

                    epochs=150,

                    validation_split = 0.2)
# Calculate classification accuracy on the testing set

score = model.evaluate(x_test, y_test)

score
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("BATCH:                 " + str(200) + "\n" +

          "VALIDATION LOSS:        " + str(score[0]) + "\n" +

          "VALIDATION ACCURACY: " + str(score[1]), loc = "left")

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['test', 'train'], loc='upper left')

plt.show()
# Create a list of symbols

symbols = ['а','б','в','г','д','е','ё','ж','з','и','й',

           'к','л','м','н','о','п','р','с','т','у','ф',

           'х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']
# Model predictions for the testing dataset

y_test_predict = model.predict_classes(x_test)
# Display true labels and predictions

fig = plt.figure(figsize=(14, 14))

for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):

    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(x_test[idx]))

    pred_idx = y_test_predict[idx]

    true_idx = np.argmax(y_test[idx])

    ax.set_title("({})".format(symbols[true_idx]))
model.save('my_model.h5')
def revert_categorical(data):

    result = []

    for i in data:

       result.append(symbols[np.argmax(i)])

    return result
correct = []

wrong = []

for i, idx in enumerate(y_test):

    pred_idx = y_test_predict[i]

    true_idx = np.argmax(y_test[i])

    

    if pred_idx == true_idx:

        correct.append(symbols[true_idx])

    else:

        wrong.append(symbols[true_idx])
a = pd.DataFrame({'Letter':wrong}).Letter.value_counts().sort_index()

b = pd.DataFrame({'Letter':revert_categorical(y_test)}).Letter.value_counts().sort_index()

c = a.divide(b)

c.plot(kind='bar', figsize=(20,10), title='Wrong', fontsize=30)
pd.DataFrame({'Letter':correct}).Letter.value_counts().sort_index().plot(kind='bar', figsize=(20,10), title='Correct')
pd.DataFrame({'Letter':wrong}).Letter.value_counts().sort_index().plot(kind='bar', figsize=(20,10), title='Wrong', fontsize=30)
def plot_categorical(data):

    result = []

    for i in data:

       result.append(symbols[np.argmax(i)])

    df = pd.DataFrame({'Letter':result})

    df.Letter.value_counts().sort_index().plot(kind='bar', figsize=(20,10), fontsize=30)
plot_categorical(cat_targets)
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



image_size = 32



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=50,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.5, # Randomly zoom image 

        width_shift_range=0.4,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.4,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True)  # randomly flip images





datagen.fit(x_train)
fig = plt.figure(figsize=(14, 14))

# configure batch size and retrieve one batch of images

for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=16):

	# create a grid of 3x3 images

	for i in range(0, 9):

		plt.subplot(330 + 1 + i)

		plt.imshow(X_batch[i])

	# show the plot

	plt.show()

	break
# Display true labels and predictions

fig = plt.figure(figsize=(14, 14))

for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):

    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(x_test[idx]))

    pred_idx = y_test_predict[idx]

    true_idx = np.argmax(y_test[idx])

    ax.set_title("{} ({})".format(symbols[pred_idx], symbols[true_idx]),

                 color=("#4876ff" if pred_idx == true_idx else "darkred"))