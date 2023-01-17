import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')

test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')

train.head()
train.shape
from IPython.display import Image

Image("../input/sign-language-mnist/american_sign_language.PNG")
Image("../input/sign-language-mnist/amer_sign2.png")
labels = train['label'].values

unique_val = np.array(labels)

np.unique(unique_val)
plt.figure(figsize = (18,8))

sns.countplot(x =labels)
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)

labels
train.drop('label', axis = 1, inplace = True)

images = train.values
plt.style.use('grayscale')

fig, axs = plt.subplots(1, 5, figsize=(15, 4), sharey=True)

for i in range(5): 

        axs[i].imshow(images[i].reshape(28,28))

fig.suptitle('Grayscale images')
images =  images/255
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.4, stratify = labels, random_state = 7)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
num_classes = 24

batch_size = 150

epochs = 50
model = Sequential()

model.add(Conv2D(64, kernel_size=(4,4), dilation_rate=(2,2), activation = 'relu', input_shape=(28, 28 ,1), padding='same' ))

model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size = (4, 4), activation = 'relu', padding='same' ))

model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size = (2, 2)))





model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer='nadam',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(shear_range = 0.25,

                                   zoom_range = 0.15,

                                   rotation_range = 15,

                                   brightness_range = [0.15, 1.15],

                                   width_shift_range = [-2,-1, 0, +1, +2],

                                   height_shift_range = [ -1, 0, +1],

                                   fill_mode = 'reflect')

test_datagen = ImageDataGenerator()
history1 = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
plt.style.use('ggplot')

plt.plot(history1.history['accuracy'])

plt.plot(history1.history['val_accuracy'])

plt.ylim(0.80, 1.05)

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])

plt.show()
test_labels = test['label']

test.drop('label', axis = 1, inplace = True)

test_images = test.values/255

test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])

test_images = np.array([i.flatten() for i in test_images])

test_labels = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

test_images.shape
# predictions

y_pred = model.predict(test_images)

from sklearn.metrics import accuracy_score

y_pred = y_pred.round()

accuracy_score(test_labels, y_pred)
model.predict(x_test[:1])
test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')

print(test)
from sklearn.metrics import confusion_matrix

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(10,10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=5)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

y_pred = model.predict(x_test)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(24))

                                                    
from PIL import Image

im = Image.open('../input/test-data-02/greyscale_01.png','r')

pix_val = list(im.getdata())

pix_val_flat = [x for sets in pix_val for x in sets]
pic_arr=[]

for i in range(len(pix_val_flat)):

    if pix_val_flat[i]!=255:

        pic_arr.append(pix_val_flat[i])

for i in range(len(pic_arr)):

    pic_arr[i]/=255

print(x_test[:1].shape)
import numpy

arr = numpy.array(pic_arr)

arr = arr.reshape(1, 28, 28, 1)

arr.shape

print(arr)
model.predict(arr)