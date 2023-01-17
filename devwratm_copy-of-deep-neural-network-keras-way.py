# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv');

test_images = (pd.read_csv('../input/test.csv').values).astype('float32')

train_images = (train.ix[:, 1:].values).astype('float32');

train_labels = (train.ix[:, 0].values).astype('int32')
train_images.shape
train_images = train_images.reshape(train_images.shape[0], 28, 28)
plt.subplot(337)

plt.imshow(train_images[6], cmap=plt.get_cmap('gray'))
test_images = test_images.reshape(test_images.shape[0], 28, 28)
plt.subplot()

plt.imshow(test_images[0], cmap=plt.get_cmap('gray'))
train_labels
train_labels[23]
train_images = train_images.reshape(42000, 28*28)

train_images.shape
test_images = test_images.reshape(test_images.shape[0], 28*28)

test_images.shape
train_images = train_images / 255;

test_images = test_images / 255;
from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_labels)
train_labels.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout

seed = 43

np.random.seed(seed)
model = Sequential()

model.add(Dense(32, activation='relu', input_dim=(28*28)))

model.add(Dense(16, activation='relu'))

model.add(Dense(10, activation='softmax'))
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),

             loss='categorical_crossentropy',

             metrics=['accuracy'])
history = model.fit(train_images, train_labels, validation_split=0.05, nb_epoch=25, batch_size=64)
history.history
history_dict = history.history

history_dict.keys()
loss_values = history_dict['loss']

acc_values = history_dict['acc']

val_loss_values = history_dict['val_loss']

val_acc_values = history_dict['val_acc']
epochs = range(1, 1 + len(loss_values))
plt.plot(epochs, loss_values, 'bo')

plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
predictions = model.predict_classes(test_images)
submission = pd.DataFrame({'ImageId': list(range(1, len(predictions)+1)),

                          "Label": predictions})

submission.to_csv("DR.csv", index=False, header=True)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("Devwrat.csv", index=False, header=True)
print(check_output(["ls", "-R", ".."]).decode("utf8"))

submission