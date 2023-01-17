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
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

X_train = df_train.drop(columns='label')
X_test = df_test.copy()

y_train = df_train['label']
Y_train = pd.get_dummies(y_train)
img = X_train.iloc[0].values.reshape((28, 28))
plt.imshow(img, cmap='gray');
def calc_hog_features(X, image_shape=(28, 28), pixels_per_cell=(8, 8)):
    fd_list = []
    for row in X:
        img = row.reshape(image_shape)
        fd = hog(img, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1))
        fd_list.append(fd)
    
    return np.array(fd_list)
X_train = calc_hog_features(X_train.values, pixels_per_cell=(8, 8))
X_test = calc_hog_features(X_test.values, pixels_per_cell=(8, 8))
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.preprocessing import normalize
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
model = Sequential()
model.add(Dense(100, input_dim=72, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
X_train_normalized = normalize(X_train)
history = model.fit(X_train_normalized, Y_train, epochs=20,
                    batch_size=5, verbose=1)
X_test_normalized = normalize(X_test)
y_pred = model.predict_classes(X_test_normalized)
df_submission = pd.DataFrame(y_pred, columns=['Label'])
df_submission.insert(0, 'ImageId', range(1, 1 + len(y_pred)))
df_submission.to_csv('submission.csv', index=False)
df_submission.head()