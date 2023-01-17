# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras.utils import to_categorical

from keras.models import Sequential, load_model

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam,RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import confusion_matrix
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.shape
train.head()
X = train.drop('label', axis=1)

y = train['label']
test.shape
train['label'].value_counts()
sns.countplot(train['label'])
X.shape
first_row = X.iloc[0].copy()
first_mat = first_row.values.reshape(28,28)
plt.imshow(first_mat)
plt.figure(figsize=(15,10))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X.iloc[i].values.reshape(28,28))

    plt.xlabel(y[i])
input_shape = (28, 28, 1)
unique_labels = y.unique()
unique_labels
num_labels = len(unique_labels)
num_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_train.shape
X_train.shape
X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
X_train.shape
X_train = X_train / 255.

X_test = X_test / 255.
###  Model Definition

model = Sequential()



# add 32 convolution filters used each of size 5x5 with relu activation

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Valid', activation='relu', input_shape=(28, 28, 1)))





# add another 32 convolution filters used each of size 3x3 with relu activation

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))



# adding pooling layer with a MaxPool2D filter of size 2x2 summarize the presence of features

# in patches of the feature map.

model.add(MaxPool2D(pool_size=(2, 2)))





# turn on and off neurons randomly for reducing interdependent learning amongst the neurons.

model.add(Dropout(0.2))



# add 64 convolution filters used each of size 5x5 with relu activation

model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Valid', activation='relu'))



# add 64 convolution filters used each of size 3x3 with relu activation

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))



# adding pooling layer with a MaxPool2D filter of size 2x2 summarize the presence of features

# in patches of the feature map.

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))



# turn on and off neurons randomly for reducing interdependent learning amongst the neurons.

model.add(Dropout(0.2))



# # Flattens the data.

model.add(Flatten())



# add densely-connected NN layer, to fully connected to drives the final classification decision.

model.add(Dense(519, activation="relu"))



# turn on and off neurons randomly for reducing interdependent learning amongst the neurons.

model.add(Dropout(0.5))



# output a softmax to let the output to be interpreted as probabilities

model.add(Dense(10, activation="softmax"))

model.summary()
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
model.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.RMSprop(),

              metrics=['accuracy'])
y_train = to_categorical(y_train, num_classes = num_labels)

y_test = to_categorical(y_test, num_classes = num_labels)


reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=2, factor=0.5, min_lr=0.0000001)
img_data_gen = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=False,

    vertical_flip=False)



# epochs     - One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE

num_epochs = 1 # replace it to 30

# batch size -Total number of training examples present in a single batch.

batch_size = 64



train_generator = img_data_gen.flow(X_train, y_train, batch_size=batch_size)

test_generator = img_data_gen.flow(X_test, y_test, batch_size=batch_size)
# Save the model to disk

model.save('MNIST-1.h5')





history = model.fit_generator(train_generator,

                    epochs=num_epochs,

                    validation_data=test_generator,

                    callbacks=[reduce_lr])
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1]) 
y_true =  [np.argmax(i) for i in y_test]

predictions = model.predict(X_test)

y_pred = [np.argmax(i) for i in predictions]

plt.figure(figsize=(15,8))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="coolwarm", annot=True , fmt="d")
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1])

plt.legend(loc='lower right')
predictions[0]
np.argmax(predictions[0])
np.argmax(y_test[0])
def plot_image(i, predictions_array, true_label, img):

    predictions_array, true_label, img = predictions_array, true_label[i], img[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])



    plt.imshow(img, cmap=plt.cm.binary)



    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:

        color = 'blue'

    else:

        color = 'red'



    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,

                                100*np.max(predictions_array),

                                true_label),

                                color=color)



def plot_value_array(i, predictions_array, true_label):

    predictions_array, true_label = predictions_array, true_label[i]

    plt.grid(False)

    plt.xticks(range(10))

    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color="#777777")

    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_array)



    thisplot[predicted_label].set_color('red')

    thisplot[true_label].set_color('blue')
num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

    plt.subplot(num_rows, 2*num_cols, 2*i+1)

    plot_image(i, predictions[i], np.argmax(np.array(y_test), axis=1), X_test.reshape(-1,28,28))

    plt.subplot(num_rows, 2*num_cols, 2*i+2)

    plot_value_array(i, predictions[i], np.argmax(np.array(y_test), axis=1))

plt.tight_layout()

plt.show()
errors = pd.DataFrame(np.argmax(y_test, axis=1), columns=['label'])
errors.reset_index(inplace=True)
errors
errors['predictions'] = y_pred
errors.loc[errors['label'] - errors['predictions'] != 0, 'error'] = 1
errors[errors['error']==1]
num_errors = len(errors[errors['error']==1].index)
print("number of errors is: {}".format(num_errors))
err_index = errors[errors['error']==1].index
plt.figure(figsize=(15,10))

for i in range(10):

    err_index = errors[errors['error']==1].index[i]

    plt.subplot(2,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_test[err_index].reshape(28,28))

    plt.xlabel("ture is {}, predicted as {}".format(np.argmax(y_test[err_index]), y_pred[err_index]))


test = test / 255

test = test.values.reshape(-1, 28, 28, 1)
final_predictions = model.predict(test)
final_predictions
final_predictions = list(map(lambda x : np.argmax(np.round(x)), final_predictions))
final_predictions[:10]
predicted_labels = pd.Series(final_predictions, name="Label")

image_id = pd.Series(range(1, len(predicted_labels)+1),name="ImageId")



results = pd.concat([image_id,predicted_labels],axis=1)



results.to_csv("MNIST.csv",index=False)