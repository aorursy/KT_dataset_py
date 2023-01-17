# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns
#load data 



train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



#verifying data 



print(train_data.head())

print(test_data.head())



Y_train = train_data['label']



#drop columnm named label 



X_train = train_data.drop(labels = ['label'], axis = 1).values

print(X_train)

print(Y_train)



X_test = test_data.values



print(X_test)
#checking the counts for each label 

sns.countplot(Y_train)



print(Y_train.value_counts().sort_index())

#making plot 

plt.figure(figsize=(15,6))

for i in range(30):

    plt.subplot(3,10,i+1)

    plt.imshow(X_train[i].reshape((28,28)),cmap = plt.cm.binary)

    plt.title("label= %d" % Y_train[i], y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.2,hspace= 0.1)

plt.show()
from sklearn.model_selection import train_test_split

X, X_test_1, Y, Y_test_1 = train_test_split(X_train, Y_train, test_size = 0.2)



print(X.shape, Y.shape)

print(X_test_1.shape, Y_test_1.shape)
#reshape flattened data into 3D tensor & normalize the value in the dataset by dividingby 255



n_x = 28



train_img = X.reshape((-1,n_x,n_x,1)).astype('float32')/255

dev_img = X_test_1.reshape((-1,n_x,n_x,1)).astype('float32')/255

test_img = X_test.reshape((-1,n_x,n_x,1)).astype('float32')/255





print(train_img.shape, dev_img.shape, test_img.shape)





#encode the labels in Y, Y_test_1



from keras.utils.np_utils import to_categorical 



train_label = to_categorical(Y)

dev_label = to_categorical(Y_test_1)

from keras_preprocessing.image  import ImageDataGenerator 

data_augment = ImageDataGenerator(rotation_range = 10, zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1)







from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Conv2D(16, kernel_size = 3, padding ='same', activation = 'relu', input_shape=(28,28,1)))

model.add(layers.Conv2D(32, kernel_size = 5, padding = 'same', activation = 'relu', input_shape = (28,28,1)))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Dropout(rate = 0.4))

model.add(layers.Conv2D(64, kernel_size = 5, activation = 'relu'))

model.add(layers.MaxPooling2D(pool_size = (2,2)))

model.add(layers.Dropout(rate = 0.4))

model.add(layers.Conv2D(128, kernel_size = 3, activation = 'relu'))

model.add(layers.Flatten())

model.add(layers.Dense(256,activation = 'relu'))

model.add(layers.Dropout(rate= 0.4))

model.add(layers.Dense(10,activation = 'softmax'))

model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
epochs = 5

batch_size = 64

log = model.fit_generator(data_augment.flow(train_img, train_label, batch_size = batch_size),epochs=epochs, steps_per_epoch=train_img.shape[0]//batch_size,validation_data=(dev_img, dev_label) )
#predict on test set 



predictions = model.predict(test_img)

print(predictions.shape)
# set the predicted labels to be the one with the highest probability

predicted_labels = []

for i in range(len(predictions)):

    predicted_label = np.argmax(predictions[i])

    predicted_labels.append(predicted_label)



# look at some of the predictions for test_X

plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(test_img[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("predict=%d" % predicted_labels[i],y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# creating submission file

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')



# generate submission file





submission['Label'] = predicted_labels



submission.to_csv('submission.csv', index=False)



print(submission)