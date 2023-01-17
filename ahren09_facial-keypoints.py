import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

TRAIN = '/kaggle/input/facial-keypoints-detection/training/training.csv'

TEST = '/kaggle/input/facial-keypoints-detection/test/test.csv'

LOOKUP = '/kaggle/input/facial-keypoints-detection/IdLookupTable.csv'

    

train, test, lookup = pd.read_csv(TRAIN), pd.read_csv(TEST), pd.read_csv(LOOKUP)
# Show one example

W, H = 96, 96

samples = train.head(4)

img0_data = samples['Image'][0]

img0_train = samples.drop(['Image'], axis=1)

img0_train = img0_train.iloc[0]

img0_X = np.array(img0_data.split(' '), dtype='float').reshape(W,H)

img0 = img0_X/255

plt.imshow(img0, cmap='gray')

plt.show()
# Fill NaN values

train.isnull().any().value_counts()
train.fillna(method='ffill', inplace=True)
# Split dataset into X and y

X_tmp = train['Image']

y = train.drop(['Image'], axis=1)

X = []

for i in range(len(X_tmp)):

    img = np.array(X_tmp[i].split(' '), dtype='float').reshape(W,H,1)

    X.append(img)

X = np.array(X)
def showKeypoints(X, y):

    img0 = X.copy().reshape(96,96)

    i = 0

    while i<30:

        x_coor, y_coor = int(y[i]), int(y[i+1])

        img0[y_coor, x_coor] = 255

        i += 2

    img0 = img0/255

    plt.imshow(img0, cmap='gray')

    plt.show()

showKeypoints(X[0], y.iloc[0])
img_sample = X[7000].reshape(W,H)/255

plt.imshow(img_sample, cmap='gray')

plt.show()
from keras.layers import Conv2D,Dropout,Dense,Flatten, BatchNormalization, MaxPool2D

from keras.models import Sequential



model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(96, 96, 1)))

model.add(BatchNormalization(momentum=0.99))

model.add(Conv2D(32, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization(momentum=0.99))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization(momentum=0.99))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization(momentum=0.99))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization(momentum=0.99))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization(momentum=0.99))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization(momentum=0.99))

model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization(momentum=0.99))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Flatten())

# model.add(Dense(1024))

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30,activation='relu'))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(X, y, epochs=200, batch_size=64, validation_split=0.2)
from keras.models import load_model

model.save('model.h5')



REUSE = False

if REUSE and os.exist('model.h5'):

    model = load_model('model.h5')
X_test_tmp = test['Image']

print(X_test_tmp.shape)

X_test = []

for i in range(len(X_test_tmp)):

    img = np.array(X_test_tmp[i].split(' '), dtype='float').reshape(W,H,1)

    X_test.append(img)

X_test = np.array(X_test)

y_pred = model.predict(X_test)

print(y_pred.shape)
# Show prediction examples

N = 3

for i in range(N):

    showKeypoints(X_test[i], pd.DataFrame(y_pred).iloc[i])
prediction = []

for col in range(len(lookup)):

    img_id = y_pred[col]['ImageId']-1

    feat = y_pred[col]['FeatureName']

    prediction[col] = y_pred[img_id][feat]
prediction = pd.DataFrame(y_pred, columns=['Location'])





prediction.insert(0, 'RowId', value=range(1, len(y_pred)+1))

prediction.to_csv('submission.csv', index=False)