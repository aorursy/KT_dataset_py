# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!unzip -u ../input/facial-keypoints-detection/test.zip

!unzip -u ../input/facial-keypoints-detection/training.zip

train_file = 'training.csv'

test_file = 'test.csv'

lookup_file = '../input/facial-keypoints-detection/IdLookupTable.csv'

train = pd.read_csv(train_file)

test = pd.read_csv(test_file)

lookup = pd.read_csv(lookup_file)

train.head().T
test.head()
lookup.head().T
train.isnull().any().value_counts()#

sns.heatmap(train.isnull(),yticklabels = False, cbar ='BuPu')

train.fillna(method = 'ffill',inplace = True)

train.isnull().any().value_counts()

train.shape

def process_img(data):

    images = []

    for idx, sample in data.iterrows():

        image = np.array(sample['Image'].split(' '), dtype=int)

        image = np.reshape(image, (96,96,1))

        images.append(image)

    images = np.array(images)/255.

    return images



def keypoints(data):

    keypoint = data.drop('Image',axis = 1)

    keypoint_features = []

    for idx, sample_keypoints in keypoint.iterrows():

        keypoint_features.append(sample_keypoints)

    keypoint_features = np.array(keypoint_features, dtype = 'float')

    return keypoint_features

y_test = test.Image

y_test.head()
X_train = process_img(train)

y_train = keypoints(train)





plt.imshow(process_img(train)[0].reshape(96,96),cmap='gray')

plt.show()


X_train.shape

# 30 metrics
import tensorflow as tf

from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Conv2D,Dropout,Dense,Flatten

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Conv2D,MaxPool2D, ZeroPadding2D

from keras.models import Sequential, Model



model = Sequential([Flatten(input_shape=(96,96)),

                         Dense(128, activation="relu"),

                         Dropout(0.1),

                         Dense(64, activation="relu"),

                         Dense(30)

                         ])
model = Sequential()



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

# model.add(BatchNormalization())

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(30))



model.compile(optimizer='adam', 

              loss='mean_squared_error',

              metrics=['mae', 'acc'])



model.summary()
model.fit(X_train,y_train,epochs = 40,batch_size = 256,validation_split = 0.2)



'''try:

    plt.plot(history.history['mae'])

    plt.plot(history.history['val_mae'])

    plt.title('Mean Absolute Error vs Epoch')

    plt.ylabel('Mean Absolute Error')

    plt.xlabel('Epochs')

    plt.legend(['train', 'validation'], loc='upper right')

    plt.show()

    

    

    # summarize history for accuracy

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Accuracy vs Epoch')

    plt.ylabel('Accuracy')

    plt.xlabel('Epochs')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Loss vs Epoch')

    plt.ylabel('Loss')

    plt.xlabel('Epochs')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

except:

    print("One of the metrics used for plotting graphs is missing! See 'model.compile()'s `metrics` argument.")'''

X_train
X_train = process_img(train)

y_test = process_img(test)

preds = model.predict(y_test)

print(preds)
def plot_sample(image, keypoint, axis, title):

    image = image.reshape(96,96)

    axis.imshow(image, cmap='gray')

    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)

    plt.title(title)
fig = plt.figure(figsize=(20,16))

for i in range(20):

    axis = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])

    plot_sample(y_test[i], preds[i], axis, "")

plt.show()
lookup.head()
print(preds[0][1])
feature = list(lookup['FeatureName'])

image_ids = list(lookup['ImageId']-1)

row_ids = lookup['RowId']

pre_list = list(preds)



feature_list = []

for f in feature:

    feature_list.append(feature.index(f))

 

final_preds = []

for x,y in zip(image_ids, feature_list):

    final_preds.append(pre_list[x][y])

    

row_ids = pd.Series(row_ids, name = 'RowId')

locations = pd.Series(final_preds, name = 'Location')

locations = locations.clip(0.0,96.0)

submission_result = pd.concat([row_ids,locations],axis = 1)

submission_result.to_csv('submission.csv',index = False)

submission_result