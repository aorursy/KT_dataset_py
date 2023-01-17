    # This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# create the training & test sets, skipping the header row with [1:]

train = pd.read_csv("../input/train.csv")



test_images = (pd.read_csv("../input/test.csv").values).astype('float32')
train_images = (train.ix[:,1:].values).astype('float32')

train_labels = train.ix[:,0].values.astype('int32')
train_images.shape
#Convert train datset to (num_images, img_rows, img_cols) format 



train_images = train_images.reshape(train_images.shape[0],  28, 28)



for i in range(6, 9):

    plt.subplot((i+1)+330)

    plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))

    plt.title(train_labels[i]);
train_images = train_images.reshape((42000, 28 * 28))



test_images.shape
train_labels.shape
train_labels
train_images = train_images / 255

test_images = test_images / 255
from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_labels)

num_classes = train_labels.shape[1]

num_classes
plt.title(train_labels[9])

plt.plot(train_labels[9])

plt.xticks(range(10));
# fix random seed for reproducibility

seed = 43

np.random.seed(seed)
train_images.shape
train_labels.shape
from keras.models import Sequential

from keras.layers import Dense , Dropout



model=Sequential()

model.add(Dense(32,activation='relu',input_dim=(28 * 28)))

model.add(Dense(16,activation='relu'))

model.add(Dense(10,activation='softmax'))
from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),

 loss='categorical_crossentropy',

 metrics=['accuracy'])
history=model.fit(train_images, train_labels, validation_split = 0.05, 

            epochs=25, batch_size=64)
history_dict = history.history

history_dict.keys()
import matplotlib.pyplot as plt

%matplotlib inline

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss_values, 'bo')

# b+ is for "blue crosses"

plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.show()
plt.clf()   # clear figure

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']



plt.plot(epochs, acc_values, 'bo')

plt.plot(epochs, val_acc_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.show()
model = Sequential()

model.add(Dense(64, activation='relu',input_dim=(28 * 28)))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(10, activation='softmax'))





model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',

 metrics=['accuracy'])



history=model.fit(train_images, train_labels, 

            epochs=15, batch_size=64)
predictions = model.predict_classes(test_images, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)
model=Sequential()

model.add(Dense(128,activation='relu',input_dim=(28*28)))

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))



model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])



history=model.fit(train_images,train_labels,epochs=20,batch_size=128)



history_dict=history.history

history_dict.keys()
import matplotlib.pyplot as plt

%matplotlib inline

loss_values = history_dict['loss']

val_acc = history_dict['acc']

epochs = range(1, len(loss_values) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss_values, 'bo')

# b+ is for "blue crosses"

plt.plot(epochs, val_acc, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.show()
predictions = model.predict_classes(test_images, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)