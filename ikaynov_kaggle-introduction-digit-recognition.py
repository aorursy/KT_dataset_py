# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import random

# reformat target vector from categorical label to one-hot-encoding
from tensorflow.keras import utils

from sklearn.model_selection import train_test_split
# Load the data
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
print('Training set size', len(train))
print('Test set size', len(test))

y_train = train["label"].to_numpy()

# Drop 'label' column
x_train = train.drop(labels = ["label"],axis = 1).to_numpy()
x_test = test.to_numpy()
print('Shape of the train data', x_train.shape)
print('Shape of the test data', x_test.shape)


# free some space
del train
# Display how many of the training samples you have per digit
g = sns.countplot(y_train)
# plot a sample point
plt.imshow(x_train[random.randint(0, x_train.shape[-1]), :].reshape(28, 28), cmap='binary') # Note that you need to reshape the data before you can plot it.
# plot first few numbers
nrows, ncols = 8, 12
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
axs = axs.ravel() 
for i in range(nrows*ncols):
    axs[i].imshow(x_train[i].reshape(28, 28), cmap='binary')
    axs[i].set(xticks=[], yticks=[])
# Check the data
X_train.isnull().any().describe()
test.isnull().any().describe()
# scale data
x_train, x_val = x_train / 255.0, x_val / 255.0

# inspect shape and type
x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_val.dtype, y_train.dtype
# reformat labels to one-hot-encoded labels
print('Before y_train[0] = {}'.format(y_train[0]))
y_train = utils.to_categorical(y_train, 10)
print('After y_train[0] = {}'.format(y_train[0]))
# Split the train and the validation set for the fitting
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2020)
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# define model topology
model = models.Sequential()
model.add(layers.Dense(40, activation='relu', input_shape=(784,)))
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# define model optimization method
model.compile(optimizer=optimizers.Adam(lr=0.001), 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])
# train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=60, validation_data=(x_val, y_val))
sns.lineplot(data=pd.DataFrame(history.history))
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")
# In the submission example we start counting from 1.
submission = pd.concat([pd.Series(range(1,28001), name="ImageId"), results], axis=1)

submission.to_csv("mlp_mnist_submission.csv", index=False)