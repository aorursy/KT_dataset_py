# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,MaxPooling2D
import keras.backend as K
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_images = np.load('../input/kmnist-train-imgs.npz')['arr_0']
test_images = np.load('../input/kmnist-test-imgs.npz')['arr_0']


train_labels = np.load('../input/kmnist-train-labels.npz')['arr_0']
test_labels = np.load('../input/kmnist-test-labels.npz')['arr_0']
print("K-MNIST train shape:", train_images.shape)
print("K-MNIST test shape:", test_images.shape)

print("K-MNIST train shape:", train_labels.shape)
print("K-MNIST test shape:", test_labels.shape)
plt.figure(figsize = (10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# Sample from the training set
sample_size = 5000

idx = np.random.choice(50000, size=sample_size, replace=False)
train_sample = train_images.reshape(60000, -1)[idx, :]
label_sample = train_labels[idx]

# Generate 2D embedding with TSNE
embeddings = TSNE(verbose=2).fit_transform(train_sample)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

plt.figure(figsize=(20,10))
plt.scatter(vis_x, vis_y, c=label_sample, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.show()
x_train_flat = train_images.reshape(60000,-1)

x_test_flat = test_images.reshape(10000,-1)
x_train = np.expand_dims(train_images,axis = -1)
x_test = np.expand_dims(test_images,axis= -1)

y_train = keras.utils.to_categorical(train_labels)
y_test = keras.utils.to_categorical(test_labels)
img_rows , img_cols = 28,28
batch_size = 128 
num_classes = 10 
epochs = 50
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, 
                             optimizer = keras.optimizers.Adam(),
                                 metrics = ['accuracy'])
history = model.fit(x_train,y_train, batch_size = batch_size, epochs =epochs, verbose = 1, validation_split = 0.3)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')
test_accuracy = model.evaluate(x_test, y_test)[1]
test_accuracy
baseline = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
baseline.fit(x_train_flat, train_labels)
print("k-NN Test Accuracy:", baseline.score(x_test_flat, test_labels))
