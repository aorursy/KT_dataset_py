# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras import layers



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import random
### import data

train_data = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv', engine = 'python')

test_data = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv', engine = 'python')

# train_data.head()
# print("train shape")

# train_data.shape

# print("test shape")

test_data.describe()
### Data assignment and normalization

image_size = 28*28

X_train = train_data.drop('label', axis =1).copy()

X_test = test_data.drop('label', axis = 1).copy()



Y_train = train_data['label'].copy()



X_train = X_train / 255.0

X_test = X_test / 255.0



# X_train.head()

# Y_train.head()
X_test.describe
### Showing Images

fig, ax = plt.subplots(figsize = (2, 2))

trash = np.asarray(X_train.iloc[[3544],:]).reshape(28, 28)

trash.shape

plt.imshow(trash)

plt.title("Sample Image")

plt.show()

X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
### data spliting and one-hot encoding

num_class = len(Y_train.unique())+1

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2)

Y_train = keras.utils.to_categorical(Y_train, num_classes = num_class)

Y_val = keras.utils.to_categorical(Y_val, num_classes = num_class)
def CNN():

    model = keras.Sequential()

    model.add(layers.Conv2D(32, (3, 3), (1, 1), padding ='valid', input_shape = (28, 28, 1), activation = 'relu'))

    model.add(layers.Conv2D(32, (3, 3), (1, 1), padding = 'same', activation = 'relu'))

    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2, 2), (2, 2), padding = 'valid'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, (3, 3), (1, 1), padding ='valid', activation = 'relu'))

    model.add(layers.Conv2D(64, (3, 3), (1, 1), padding = 'same', activation = 'relu'))

    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2, 2), (2, 2), padding = 'valid'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())

    model.add(layers.Dense(100, activation = 'relu'))

    model.add(layers.Dense(num_class, activation = 'softmax'))

    

    return model

    

    

    
model = CNN()

model.compile(optimizer = 'adam', loss = 'CategoricalCrossentropy', metrics = ['accuracy'])

model.summary()
history = model.fit(X_train, Y_train, 

                  validation_data = (X_val, Y_val),

                   batch_size = 25, epochs = 3)
plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label = 'accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend(loc = 'lower right')



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label = 'loss')

plt.plot(history.history['val_loss'], label = 'val_loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(loc = 'lower right')

plt.tight_layout()

plt.show()
# from sklearn.metrics import mean_absolute_error

# val_preds = model.predict(X_val)

# val_mae = mean_absolute_error(val_preds, Y_val)

# val_mae
def predict(model, X, imgs):

    y_test = test_data['label'].copy()

    s = int(np.sqrt(imgs))

    fig, ax = plt.subplots(s, s, sharex=True, sharey=True, figsize=(15, 15))

    ax = ax.flatten()

    preds = model.predict(X[:imgs])

    for i in range(imgs):

        y_pred = np.argmax(preds[i])

        img = X[i].reshape(28, 28)

        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

        ax[i].set_title(f'p: {y_test[i]}')
predict(model, X_test, 25)
test_Y = test_data['label'].copy()

check = test_Y[:25]

check
test_preds = model.predict(X_test)

test_preds = np.argmax(test_preds, axis = 1)

name = "Prajesh Sanghvi"

file_name = name + "_MNIST_sign_language.csv"

test_preds = pd.Series(test_preds, name = 'label')

submission = pd.concat([pd.Series(range(1,7173), name = "ImageID"), test_preds], axis = 1)

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',

          'U', 'V', 'W', 'X', 'Y']

matching = []

test_match = np.asarray(test_preds)

for i in range(len(test_match)):

    if test_match[i] < 9:

        matching.append(letters[test_match[i]])

    else:

        matching.append(letters[test_match[i]-1])

match_series = pd.Series(matching, name = 'letters')

final_sub = pd.concat([submission, match_series], axis = 1)

final_sub.to_csv(file_name, index = False)