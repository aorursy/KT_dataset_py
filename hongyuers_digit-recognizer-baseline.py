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
import pandas as pd
train = pd.read_csv('/kaggle/input/train.csv')
label = train['label']
train = train.drop(['label'], axis=1)
test = pd.read_csv('/kaggle/input/test.csv')
train.info()
train = train.values.reshape((42000, 28*28)).astype('float32')/255
test.info()
test = test.values.reshape((28000, 28*28)).astype('float32')/255
label.shape
from keras.utils import to_categorical

label = to_categorical(label)
import matplotlib.pyplot as plt

img = train[4]
img = img.reshape((28,28))
plt.imshow(img,cmap=plt.cm.binary)
plt.show()
from keras import models
from keras import layers
from keras.utils import plot_model
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.summary()
plot_model(network, show_shapes=True)
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
network.fit(train, label, epochs=5, batch_size=128, validation_split=0.1, shuffle=True)
preds = network.predict(test)
preds.shape
submission = pd.read_csv('/kaggle/input/sample_submission.csv')
submission.info()
results = np.argmax(preds,axis = 1)
submission['Label'] = results
submission.to_csv('submission.csv',index=False)
result = pd.read_csv('submission.csv')
result.head()
