# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.applications import resnet50

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras import layers

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/Kannada-MNIST/train.csv');

test_df = pd.read_csv('../input/Kannada-MNIST/test.csv');
train_df.head()
train_df.shape
def pad_with(vector, pad_width, iaxis, kwargs):

    pad_value = kwargs.get('padder', 10)

    vector[:pad_width[0]] = pad_value

    vector[-pad_width[1]:] = pad_value
def visualise_img(data):

    img = np.pad(np.reshape(np.ravel(data),(28,28)),2,pad_with,padder=0)

    img = img.astype('float32')/255

    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    print(img.shape)

    plt.imshow(img)

    plt.show()
visualise_img(train_df.iloc[4,1:])
x_train, x_validation, y_train, y_validation = train_test_split(train_df.iloc[:,1:], train_df.label, test_size=0.2, random_state=42, shuffle=True)
def df_to_imageDataset(train):

    train_data = [];

    for row in train.itertuples():

        img = np.pad(np.reshape(np.ravel(row[1:]),(28,28)),2,pad_with,padder=0)

        img = img.astype('float32')/255

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        train_data.append(img);

        

    return np.array(train_data)
train_data = df_to_imageDataset(x_train)

X_val = df_to_imageDataset(x_validation)

train_data.shape, X_val.shape
resnet_conv = resnet50.ResNet50(weights=None, include_top=False, input_shape=(32,32,3));
model = Sequential()

model.add(resnet_conv)

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()
labels = OneHotEncoder().fit_transform(np.array(y_train).reshape(-1,1));

y_val = OneHotEncoder().fit_transform(np.array(y_validation).reshape(-1,1));

pd.DataFrame(labels).head()
train_data.shape
model.fit(x=train_data,y=labels,shuffle=True, batch_size=128,epochs=32)
score = model.evaluate(X_val, y_val, batch_size=16)

score
test_df.head()
test_data = df_to_imageDataset(test_df.iloc[:,1:])

test_data.shape
result = model.predict(test_data)

result
np.argmax(result, axis=1)
index = 324

visualise_img(test_df.iloc[index,1:])

np.argmax(result[index])
my_submission = pd.DataFrame({'id': test_df.id, 'label': np.argmax(result, axis=1)})



my_submission.to_csv('sample_submission.csv', index=False)
test2_df = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv');

test2_df.head()
test2_df.shape
test2_data = df_to_imageDataset(test2_df.iloc[:,1:])

test2_data.shape
result = np.argmax(model.predict(test2_data), axis=1)

result
from sklearn.metrics import accuracy_score

print('Accuracy = ',accuracy_score(test2_df.label, result, normalize=False)/test2_df.shape[0])
index = 232

visualise_img(test2_df.iloc[index,1:]);

result[index]