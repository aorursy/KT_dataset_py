# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization

import matplotlib.pyplot as plt



# Preprocessing data

from sklearn.preprocessing import StandardScaler



# Neural Networks

import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")

data_test = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test.csv")
X_train = data_train.drop(columns=['label'])

y_train = data_train['label']



X_test = data_test.drop(columns=['label'])

y_test = data_test['label']
print(X_train.shape)

X_train.head()
def number_to_letter(number):

    return chr(ord('A')+number)
sample_images = data_train.groupby('label', group_keys=False).apply(lambda df: df.sample(1))

labels = sample_images['label']

images = sample_images.drop(columns=['label'])
rows = 4

columns = 6

fig, axs = plt.subplots(rows,columns, figsize=(12,12))



for i in range(rows):

    for j in range(columns):

        image_array = images.iloc[i*columns+j].values.reshape(28,28)

        axs[i][j].imshow(image_array, cmap='gray')

        axs[i][j].set_title(number_to_letter(labels.iloc[i*columns+j]))

        axs[i][j].axis('off')
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
# Reshape image (h=28px, w=28px, canal=1)

X_train_reshaped = X_train_scaled.reshape(-1,28,28,1)

X_test_reshaped = X_test_scaled.reshape(-1,28,28,1)
# WARNING: the actual num_classes should be 24, but class 9 is missing.

# Therefore the number of classes was increased to 25.

num_classes = 25

y_train_categorical = to_categorical(y_train, num_classes)

y_test_categorical = to_categorical(y_test, num_classes)
batch_size = 1000

epochs = 4
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
model.fit(X_train_reshaped, y_train_categorical,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1)
predictions = model.predict(X_test_reshaped)

score = model.evaluate(X_test_reshaped, y_test_categorical, verbose=0)

print('Test loss:{}, accuracy:{}'.format(score[0], score[1]))
preds_labels = pd.Series([p.argmax() for p in predictions])



# Series containing only the mislabeled elements

failed_preds = preds_labels[preds_labels != y_test]
rows = 4

columns = 6

fig, axs = plt.subplots(rows,columns, figsize=(12,12))



for i in range(rows):

    for j in range(columns):

        index = failed_preds.index[i*columns+j]

        image_array = X_test.loc[index].values.reshape(28,28)

        axs[i][j].imshow(image_array, cmap='gray')

        axs[i][j].set_title('Label: {}\n Predicted: {}'.format(number_to_letter(y_test.loc[index]),

                                                               number_to_letter(predictions[index].argmax())))

        axs[i][j].axis('off')