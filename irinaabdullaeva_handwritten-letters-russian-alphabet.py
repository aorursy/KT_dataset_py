# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from __future__ import division, print_function



import h5py



# Импортируем TensorFlow и tf.keras

import tensorflow as tf

from tensorflow import keras



import warnings

warnings.filterwarnings('ignore')



#%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore')

sns.set(rc={'figure.figsize' : (12, 6)})

sns.set_style("darkgrid", {'axes.grid' : True})



import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score



# Display all columns of dataframe

pd.set_option('display.max_columns', None)
data1 = pd.read_csv("../input/letters.csv")

data2 = pd.read_csv("../input/letters2.csv")

data3 = pd.read_csv("../input/letters3.csv")

print("First dataset shape: {0}, \nSecond dataset shape: {1}, \nThird dataset shape: {2}".format(data1.shape, data2.shape, data3.shape))
data1.head()
# Read the h5 file

f = h5py.File('../input/LetterColorImages_123.h5', 'r')

# List all groups

keys = list(f.keys())

keys 
# Create tensors and targets of images

img_backgrounds = np.array(f[keys[0]])

img_tensors = np.array(f[keys[1]])

targets = np.array(f[keys[2]])

print ('Tensor shape:', img_tensors.shape)

print ('Target shape', targets.shape)

print ('Background shape:', img_backgrounds.shape)
# Concatenate series

letters = pd.concat((data1["letter"], data2["letter"]), axis=0, ignore_index=True)

letters = pd.concat((letters, data3["letter"]), axis=0, ignore_index=True)

len(letters)
# Normalize the tensors

img_tensors = img_tensors/255

img_tensors[0][0][0][0]
sns.countplot(x="label", data=data1)
sns.countplot(x="background", data=data1)
# Read and display a tensor using Matplotlib

sns.set_style("darkgrid", {'axes.grid' : False})

print('Label: ', letters[100])

plt.figure(figsize=(3,3))

plt.imshow(img_tensors[100]);
type(img_tensors[0])
img_tensors[0].shape
# Display the first image of each label.

def display_images_and_labels(images, labels):

    unique_labels = set(labels)

    plt.figure(figsize=(15, 15))

    i = 1

    labels = labels.tolist()

    for label in unique_labels:

        # Pick the first image for each label.

        image = images[labels.index(label)]

        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns

        plt.axis('off')

        plt.title("Label {0} ({1})".format(label, labels.count(label)))

        i += 1

        _ = plt.imshow(image)

    plt.show()

display_images_and_labels(img_tensors, targets)
# Display images of a specific label.

def display_label_images(images, labels, label):

    limit = 24  # show a max of 24 images

    plt.figure(figsize=(15, 5))

    i = 1

    labels = labels.tolist()

    start = labels.index(label)

    end = start + labels.count(label)

    for image in images[start:end][:limit]:

        plt.subplot(3, 8, i)  # 3 rows, 8 per row

        plt.axis('off')

        i += 1

        plt.imshow(image)

    plt.show()



display_label_images(img_tensors, targets, 21)
# Display images of a specific label.

def display_images_grayscale(images, labels, label):

    limit = 24  # show a max of 24 images

    plt.figure(figsize=(15, 5))

    i = 1

    labels = labels.tolist()

    start = labels.index(label)

    end = start + labels.count(label)

    for image in images[start:end][:limit]:

        plt.subplot(3, 8, i)  # 3 rows, 8 per row

        plt.axis('off')

        i += 1

        plt.imshow(image)

    plt.show()
# Make dictionary to decode index to letters

dictionary = {'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], \

              'letter': ['а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р','с', \

                         'т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']}

letter_dict = pd.DataFrame.from_dict(dictionary)

letter_dict = letter_dict.set_index("num")

letter_dict.head()
# One-hot encoding the targets, started from the zero label

from keras.utils import to_categorical

coded_targets = to_categorical(np.array(targets-1), 33)

coded_targets.shape
from tensorflow import image

from tensorflow.image import rgb_to_grayscale

img_tensors_gs_tf = rgb_to_grayscale(img_tensors)



sess = tf.Session()

with sess.as_default():

    print(img_tensors_gs_tf.eval().shape)

    arr_img_tensors_gs_tf = img_tensors_gs_tf.eval()
img_tensors_gs = arr_img_tensors_gs_tf

for image in img_tensors_gs[:5]:

    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
# from skimage.color import rgb2grey

# # img_tensors_gs = np.asarray(images32)

# img_tensors_gs = rgb2grey(img_tensors)

# for image in img_tensors_gs[:5]:

#     print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
img_tensors_gs.shape
# Split the data to test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(img_tensors_gs, coded_targets, test_size = 0.2, random_state = 1)

print("Train dataset shape: {0}, \nTest dataset shape: {1}".format(X_train.shape, X_test.shape))
# Split the test data to validation and test sets

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 1)

print("Validation dataset shape: {0}, \nTest dataset shape: {1}".format(X_valid.shape, X_test.shape))
from keras.preprocessing import image as keras_image

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.metrics import top_k_categorical_accuracy, categorical_accuracy



from keras.models import Sequential, load_model

from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D

from keras.layers.advanced_activations import PReLU, LeakyReLU, Softmax

from keras.layers import Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D



def top_3_categorical_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def model():

    model = Sequential()

    

    # Define a model architecture    

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=X_train.shape[1:]))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    

    model.add(Conv2D(64, (3, 3), padding='same'))

    model.add(LeakyReLU(alpha=0.02))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128, (3, 3), padding='same'))

    model.add(LeakyReLU(alpha=0.02))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    

    # Global max pooling is ordinary max pooling layer with pool size equals to the size of the input (minus filter size + 1, to be precise). 

    model.add(GlobalMaxPooling2D())

    

    model.add(Dense(1024))

    model.add(LeakyReLU(alpha=0.02))

    model.add(Dropout(0.2)) 

    

    model.add(Dense(33))

    model.add(Activation('softmax'))



    # Compile the model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy, top_3_categorical_accuracy])

    return model



model = model()

model.summary()
# Create callbacks

checkpointer = ModelCheckpoint(filepath='weights.best.model.hdf5', verbose=1, save_best_only=True)

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.75)

# early_stoping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

callbacks = [checkpointer, lr_reduction]                            
# Train the model

# batch size = 473 as it divides X_train size, X_test and X_valid sizes evenly

history = model.fit(X_train, y_train, epochs=200, batch_size=473, verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks)                    
# Plot the Neural network fitting history

def history_plot(fit_history, n):

    plt.figure(figsize=(18, 12))

    

    plt.subplot(211)

    plt.plot(fit_history.history['loss'][n:], color='slategray', label = 'train')

    plt.plot(fit_history.history['val_loss'][n:], color='#4876ff', label = 'valid')

    plt.xlabel("Epochs")

    plt.ylabel("Loss")

    plt.legend()

    plt.title('Loss Function');  

    

    plt.subplot(212)

    plt.plot(fit_history.history['categorical_accuracy'][n:], color='slategray', label = 'train')

    plt.plot(fit_history.history['val_categorical_accuracy'][n:], color='#4876ff', label = 'valid')

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")    

    plt.legend()

    plt.title('Accuracy');
# Plot the training history

history_plot(history, 0)
# Load the model with the best validation accuracy

model.load_weights('weights.best.model.hdf5')

# Calculate classification accuracy on the testing set

score = model.evaluate(X_test, y_test)

score
# Model predictions for the testing dataset

y_test_predict = model.predict_classes(X_test)
# Create a list of symbols

symbols = ['а','б','в','г','д','е','ё','ж','з','и','й',

           'к','л','м','н','о','п','р','с','т','у','ф',

           'х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']
# Display true labels and predictions

fig = plt.figure(figsize=(14, 14))

for i, idx in enumerate(np.random.choice(X_test.shape[0], size=16, replace=False)):

    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(X_test[idx]), cmap="gray")

    pred_idx = y_test_predict[idx]

    true_idx = np.argmax(y_test[idx])

    ax.set_title("{} ({})".format(symbols[pred_idx], symbols[true_idx]),

                 color=("#4876ff" if pred_idx == true_idx else "darkred"))
# Save model

model.save('HW_best_model.h5')