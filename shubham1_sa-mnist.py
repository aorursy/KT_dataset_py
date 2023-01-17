# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten, Activation

from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# loading train data

train = pd.read_csv('../input/train.csv')

train_x = train.ix[:,1:]

train_y = train.ix[:,0]

print(train_x.head(3))

print('shape',train_x.shape)

# loading Test data

test_x = pd.read_csv('../input/test.csv')

print(test_x.head(3))

print('shape',test_x.shape)


X_train = train_x.values.astype('float32') 

y_train = train_y.values.astype('int32') 

X_test = test_x.values.astype('float32')

print(X_train)

print(y_train)


# Convert train datset to (num_images,img_rows,img_cols) format and show images

X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(50,55):

    plt.subplot(100 + (i+1))

    plt.imshow(X_train[i])

    plt.title(y_train[i])
# expand 1 more dimension as 1 for colour channel gray

print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

print('X_train',X_train.shape)



X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

print('X_test',X_test.shape)
# Normalize train and test data using mean and sd



mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)

#print(mean_px,std_px)

X_train = np.array((X_train-mean_px) / std_px)



mean_px = X_test.mean().astype(np.float32)

std_px = X_test.std().astype(np.float32)

#print(mean_px,std_px)

X_test = np.array((X_test-mean_px) / std_px)
label_binarizer = LabelBinarizer()

y_train = label_binarizer.fit_transform(y_train)

print(y_train)


# Build the Fully Connected Neural Network in Keras Here

model = Sequential()

model.add(Flatten(input_shape=(28,28,1)))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(10))

model.add(Activation('softmax'))



model.compile('adam', 'categorical_crossentropy', ['accuracy'])

history = model.fit(X_train, y_train, nb_epoch=10, validation_split=0.2)
# Predictions

predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})



# Writing to csv file

submissions.to_csv("mnist_result.csv", index=False, header=True)