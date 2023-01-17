from IPython.display import Image

Image("../input/amer_sign2.png")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import os

import seaborn as sns



print(os.listdir("../input"))
train = pd.read_csv("../input/sign_mnist_train.csv")

test = pd.read_csv("../input/sign_mnist_test.csv")



train.head()
X = train.drop('label', axis = 1)

y = train[['label']]



X_test = test.drop('label', axis = 1)

y_test = test[['label']]



X = X.values / 255

X_test = X_test.values / 255
unique_images = y.copy()

unique_images['pic_nr'] = unique_images.index

unique_images = unique_images.groupby('label').first().reset_index()

unique_images

    
letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'

fig, axes = plt.subplots(4,6,figsize = (20,20))

axes = axes.ravel()

for index, row in unique_images.iterrows():

    axes[index].imshow(X[row['pic_nr'],:].reshape((28,28)), cmap=cm.gray)

    axes[index].set_title(letters[index], fontsize = 20)

    axes[index].axis('off')

plt.subplots_adjust(hspace=0.4)
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

lb.fit(y)

y = lb.transform(y)

y_test = lb.transform(y_test)

from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.2, random_state = 77)

X_train = X_train.reshape(X_train.shape[0], *(28,28,1))

X_validate = X_validate.reshape(X_validate.shape[0], *(28,28,1))

X_test = X_test.reshape(X_test.shape[0], *(28,28,1))

X_train.shape
X_validate.shape
X_test.shape
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten

from keras.optimizers import Adam

from keras.callbacks import TensorBoard





from numpy.random import seed

seed(77)

from tensorflow import set_random_seed

set_random_seed(77)
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(24, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr= 0.001), metrics = ['accuracy'])


history = model.fit(X_train, y_train, batch_size=512, epochs=10, verbose = 1, validation_data=(X_validate, y_validate))
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("Training and validation accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','validation'], loc = 'lower right')

plt.show()
evaluation = model.evaluate(X_test, y_test, batch_size=512, verbose=1)

print('Accuracy on Test set: ', evaluation[1])
from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(X_test)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred)

plt.figure(figsize = (20,20))

ax= plt.subplot()

sns.heatmap(cm, annot = True, fmt='g', cmap='gist_ncar', ax =ax)

plt.ylabel('Letter in the image', fontsize=14)

plt.xlabel('Predicted letter', fontsize=14)

ax.set_xticklabels(letters) 

ax.set_yticklabels(letters, rotation = 0)

plt.show()
perfect = []

mistake_detail = []

for row in range(cm.shape[0]):

    mistakes_count = 0

    for col in range(cm.shape[1]):

        if row == col: 

            continue

        if cm[row][col]>0:

            mistakes_count = mistakes_count+1

            mistake_detail.append((letters[row], letters[col], cm[row][col]))

    if mistakes_count == 0:

        perfect.append(letters[row])

print('Letters recognized without errors:', perfect)



mistakes_df = pd.DataFrame(mistake_detail, columns = ['Original', 'Predicted', 'Count'])

print('\nErrors list:')

print(mistakes_df.sort_values(by=['Count'], ascending = False).to_string(index=False))

            

        

    