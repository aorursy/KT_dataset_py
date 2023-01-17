# this is solution using cnn deep learning algorith for digit recognization problem

# dated 10th dec 2019 - hemant bhora
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline
print('train shape : ',train.shape)

print('test shape : ',test.shape)

print('sample_submission shape: ',sample_submission.shape)
train.head()
test.head()
sample_submission.head()
train_images=train.drop(['label'],axis=1)

train_label=train['label']

# geting dependent and independent features seperated
test_final=np.asarray(test)/255

# this is test set for submission pupose
X_train, X_test, y_train, y_test = train_images[:40000], train_images[40000:], train_label[:40000], train_label[40000:]

# spliting the data for train and test purpose

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
train_X=np.asarray(X_train)/255

train_y=np.asarray(y_train)

test_X=np.asarray(X_test)/255

test_y=np.asarray(y_test)

# normalising the data
plt.figure(figsize=[5,5])

plt.subplot(122)

curr_img = np.reshape(train_X[0], (28,28))

# Display the first image in testing data

plt.imshow(curr_img, cmap='gray')

plt.show()
from sklearn import preprocessing

#x = preprocessing.StandardScaler().fit(train_X).transform(train_X)

#x[0:1]
# Reshape training and testing image

train_X = train_X.reshape(-1, 28, 28, 1)

test_X = test_X.reshape(-1,28,28,1)
train_X.shape, test_X.shape
train_y.shape, test_y.shape
test_final=test_final.reshape(-1, 28, 28, 1)
test_final.shape
from tensorflow.keras import layers, models
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu')) # relu activation function to catch non-linearity

model.add(layers.Dense(10, activation='softmax')) # for output layer
model.summary()
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy', # loss function is cross entropy

              metrics=['accuracy']) # evaluation metic i took , accuracy



history = model.fit(train_X, train_y, epochs=10, 

                    validation_data=(test_X, test_y))
test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)
print(test_acc)
y_pred=model.predict(test_final)
y_pred[0]
y_index = np.argmax(y_pred,axis=1)
y_index.shape
y=np.arange(1,28001)

y
submission = pd.DataFrame({"ImageId": y,"Label": y_index},dtype='int64')
submission
submission.to_csv('cnn_submission_mnist_kaggle.csv', index=False)
# i am done here,good bye everyone. have a good day , regards