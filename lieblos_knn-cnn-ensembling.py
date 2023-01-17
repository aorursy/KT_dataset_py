# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
train_df = pd.read_csv('../input/train.csv')
display(train_df.head(2))
test_df = pd.read_csv('../input/test.csv')
display(test_df.head(2))
from keras.utils import to_categorical

y_train = to_categorical(train_df['label'].values)
X_train = train_df.drop('label', axis=1).values.reshape(train_df.shape[0], 28, 28, 1)
X_test = test_df.values.reshape(test_df.shape[0], 28, 28, 1)
import matplotlib.pyplot as plt

display(y_train.shape)
display(X_train.shape)
display(X_test.shape)
# CNN
from keras.models import Model
from keras.layers import concatenate, Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout

# feature recognition
m1_input1 = Input(shape=(28, 28, 1))
m1_conv1 = Conv2D(32, kernel_size=3, activation='relu')(m1_input1)
m1_conv2 = Conv2D(32, kernel_size=3, activation='relu')(m1_conv1)
m1_maxp1 = MaxPooling2D(pool_size=(2, 2))(m1_conv2)
m1_conv3 = Conv2D(64, kernel_size=3, activation='relu')(m1_maxp1)
m1_conv4 = Conv2D(64, kernel_size=3, activation='relu')(m1_conv3)
m1_maxp2 = MaxPooling2D(pool_size=(2, 2))(m1_conv4)
m1_flat1 = Flatten()(m1_maxp2)

# feature classification
m1_dense1 = Dense(512, activation='relu')(m1_flat1)
m1_drop3 = Dropout(0.2)(m1_dense1)
m1_output1 = Dense(10, activation='softmax')(m1_drop3)

cnn = Model(inputs=m1_input1, outputs=m1_output1)
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
# no val final run
# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
# X_train, X_val, y_train, y_val = X_train[:1000], X_val, y_train[:1000], y_val
cnn.fit(X_train, y_train, epochs=10)
knn.fit(X_train.reshape(X_train.shape[0], 784), y_train)
cnn_preds = np.argmax(cnn.predict(X_train), axis=1)
knn_preds = knn.predict(X_train.reshape(X_train.shape[0], 784))
import xgboost as xgb

X_train = pd.DataFrame({
    '1': cnn_preds,
    '2': np.argmax(knn_preds)
})
y_train = y_train

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
gbm.fit(X_train, np.argmax(y_train, axis=1))
cnn_preds = np.argmax(cnn.predict(X_test), axis=1)
knn_preds = knn.predict(X_test.reshape(X_test.shape[0], 784))

X_test = pd.DataFrame({
    '1': cnn_preds,
    '2': np.argmax(knn_preds),
})

preds = gbm.predict(X_test)
sample_submission = pd.read_csv('../input/sample_submission.csv')
submission = pd.DataFrame({
    'ImageId': list(range(1, len(preds) + 1)),
    'Label': preds,
})
display(submission.head())
submission.to_csv('submission.csv', index=False)
