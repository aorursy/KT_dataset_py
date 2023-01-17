# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import Adam
train_df = pd.read_csv('../input/train.csv')



y_train = train_df["label"]

x_train = train_df.drop(labels = ["label"],axis = 1)

del train_df



n_train = len(x_train)

n_pixels = len(x_train.columns)

n_class = len(set(y_train))

print('Number of training samples: {0}'.format(n_train))

print('Number of training pixels: {0}'.format(n_pixels))

print('Number of classes: {0}'.format(n_class))
x_test = pd.read_csv('../input/test.csv')



n_test = len(x_test)

n_pixels = len(x_test.columns)



print('Number of train samples: {0}'.format(n_test))

print('Number of test pixels: {0}'.format(n_pixels))
x_train = x_train / 255.0

x_test = x_test / 255.0
x_train = x_train.values.reshape(-1,28,28,1)

X = x_train

x_test = x_test.values.reshape(-1,28,28,1)



#Converting to one-hot vector

y_train = to_categorical(y_train, num_classes = 10)

#g = plt.imshow(x_train[0][:,:,0])
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = 3,padding = 'Same', activation ='relu', input_shape = (28,28,1)))

#model.add(Conv2D(32, kernel_size = 5, activation='relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 5, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(48, kernel_size = 5, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Flatten())





model.add(Dense(512, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(10, activation='softmax'))

optimizer = Adam(lr=0.001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
# CREATE MORE IMAGES VIA DATA AUGMENTATION

datagen = ImageDataGenerator( rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)



#Reduce Learning rate by factor of 0.3

learning_rate_decay = ReduceLROnPlateau(monitor='acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.3, 

                                            min_lr=1e-6)
batch_size = 64

epochs = 60

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size), 

                    epochs = epochs, steps_per_epoch= x_train.shape[0] // batch_size, verbose = 2, 

                             callbacks=[learning_rate_decay])
plt.plot(history.history['acc'],linestyle='-')

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend('Train', loc='upper left')

axes = plt.gca()

axes.set_ylim([0.98,1])

plt.show()
# predict results

results = model.predict(x_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission_cnn_mnist.csv",index=False)
results.head()
import shap



# select a set of background examples to take an expectation over

background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]



# explain predictions of the model on four images

e = shap.DeepExplainer(model, background)

# ...or pass tensors directly

# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)

shap_values = e.shap_values(x_test[1:5])



# plot the feature attributions

shap.image_plot(shap_values, -x_test[1:5])