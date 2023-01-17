# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/data.csv")
dataset.head()
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
num_pixels = x.shape[1]
num_classes = 46
photo_width = 32
photo_height = 32
photo_depth = 1
x_images = x.values.reshape(x.shape[0], photo_width, photo_height)
for i in range(1,9):
    plt.subplot(240+i)
    plt.axis('off')
    plt.imshow(x_images[i-1], cmap=plt.get_cmap('gray'))
plt.show()
value_image = pd.DataFrame(dataset.iloc[:,1024].value_counts()).T.rename(index = {0:'null values (nb)'})
value_image
row_to_remove = np.where(dataset.iloc[:,1024].values == 1024)
row_to_remove
plt.imshow(x_images[2000], cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()
x = dataset.iloc[:,:-1]
X_images = x.values.reshape(x.shape[0], photo_width,photo_height)
y = dataset.iloc[:,-1]
# output in binary format
from sklearn.preprocessing import LabelBinarizer
binencoder = LabelBinarizer()
Y = binencoder.fit_transform(y)
x = x/255
seed = 123 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed)
from keras.models import Sequential, Model
from keras.layers import Dense
def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = baseline_model()

epochs = 10
batch_size = 400
history = model.fit(X_train.values, y_train, validation_split=0.20, epochs=epochs, batch_size=batch_size, verbose=2)
