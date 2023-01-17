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
#importing data from csv file and creating dataframes

kannada_test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

kannada_train_data =pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

kannada_dig_data = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')



#verifying data 

print(kannada_test_data.head())

print(kannada_train_data.head())

print(kannada_dig_data.head())



#printing the shapes of the dataframes 

print(kannada_test_data.shape, kannada_train_data.shape, kannada_dig_data.shape)

kannada_train_data['label'].value_counts().sort_index()
kannada_dig_data['label'].value_counts().sort_index()
X = kannada_train_data.drop(['label'], axis=1).values

Y = kannada_train_data['label']

X_test = kannada_test_data.drop(['id'], axis=1).values

X_test_1 = kannada_dig_data.drop(['label'], axis=1).values

Y_test_1 = kannada_dig_data['label']



#verifying data frame sizes 

print(X.shape, Y.shape, X_test.shape, X_test_1.shape, Y_test_1.shape)
import matplotlib.pyplot as plt



# look at some of the digits from train dataset

plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("label=%d" % Y[i],y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_devtest, Y_train, Y_devtest = train_test_split(X, Y, test_size = 0.2)



print(X_train.shape, Y_train.shape)

print(X_devtest.shape, Y_devtest.shape)
#preparing data for CNN



#reshape flattend data into 3D tensor & standarize the value in the dataset by dividing by 255



n_x = 28 

train_img = X_train.reshape((-1,n_x,n_x, 1)).astype('float32')/255

dev_img = X_devtest.reshape((-1,n_x,n_x,1)).astype('float32')/255

test_img = X_test.reshape((-1,n_x,n_x,1)).astype('float32')/255

dig_img = X_test_1.reshape((-1,n_x,n_x,1)).astype('float32')/255

print(train_img.shape, dev_img.shape, test_img.shape, dig_img.shape)



#one-hot encode the labels in Y_train, Y_devtest, Y_test_1



from keras.utils.np_utils import to_categorical

train_labels = to_categorical(Y_train)

dev_labels = to_categorical(Y_devtest)

dig_labels = to_categorical(Y_test_1)



print(train_labels.shape, dev_labels.shape, dig_labels.shape)

print(Y_test_1[8],dig_labels[8])



plt.figure(figsize=(1,1))

plt.imshow(X_test_1[8].reshape((28,28)), cmap=plt.cm.binary)

plt.show()

#use keras data generator to augment the training set



from keras_preprocessing.image import ImageDataGenerator

data_augment = ImageDataGenerator(rotation_range =10, zoom_range =0.1, width_shift_range =0.1, height_shift_range = 0.1)
from keras import models 

from keras import layers



model = models.Sequential()

model.add(layers.Conv2D(16, kernel_size=3, padding ='same', activation = 'relu', input_shape=(28,28,1)))

model.add(layers.Conv2D(32, kernel_size =5, padding= 'same', activation = 'relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(64, kernel_size = 5, activation= 'relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(128,kernel_size=3,activation = 'relu'))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation = 'relu'))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Dense(10,activation = 'softmax'))

model.summary()
#compiling model 



model.compile(optimizer ='adam', loss='categorical_crossentropy',metrics = ['accuracy'])


epochs = 30

batch_size = 64

history = model.fit_generator(data_augment.flow(train_img, train_labels, batch_size = batch_size),epochs=epochs, steps_per_epoch=train_img.shape[0]//batch_size,validation_data=(dev_img, dev_labels) )
#Error Analysis 



pred_dig = model.predict(dig_img)

pred_dig_labels = []

for i in range(len(dig_img)):

    pred_dig_label = np.argmax(pred_dig[i])

    pred_dig_labels.append(pred_dig_label)
#finding the numbers which were right while prediction 



result = pd.DataFrame(Y_test_1)

result['Y_pred'] = pred_dig_labels

result['correct'] = result['label'] - result['Y_pred']

error = result[result['correct'] != 0]

error_list = error.index



print('Number of error is :', len(error))

print('The incidents are : ', error_list)
# predict on test set

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

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
# generate submission file





submission['label'] = predicted_labels



submission.to_csv('submission.csv', index=False)



print(submission)