# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import keras 
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
df_train = pd.read_csv("/kaggle/input/sign_mnist_train.csv")
df_test = pd.read_csv("/kaggle/input/sign_mnist_test.csv")
print(df_train.shape)
print(df_test.shape)
(df_train['label'].unique())
df_train.head()
df_test.head()
plt.figure(figsize=(11,5))
sns.countplot(df_train['label'])
y = df_train['label'].values
x = df_train.values
df_train.drop('label', axis = 1, inplace = True)
# Label binarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(y)
labels.shape
x = df_train.values
x = np.array([np.reshape(i, (28, 28)) for i in x])
x = np.array([i.flatten() for i in x])
plt.imshow(x[2].reshape(28,28))
plt.imshow(x[2].reshape(28,28))
print(labels[2])
# letter C
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size = 0.3, random_state = 101)

x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train.shape
#learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
model = Sequential()
model.add(Conv2D(64, (3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(24, activation = 'softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=50, batch_size=128)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()
test_labels = df_test['label']
df_test.drop('label', axis = 1, inplace = True)
test_images = df_test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
y_pred = model.predict(test_images)
accuracy_score(test_labels, y_pred.round())
