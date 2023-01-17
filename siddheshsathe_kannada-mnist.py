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
train_csv = '/kaggle/input/Kannada-MNIST/train.csv'

test_csv = '/kaggle/input/Kannada-MNIST/test.csv'

val_csv = '/kaggle/input/Kannada-MNIST/Dig-MNIST.csv'
train_df = pd.read_csv(train_csv)

test_df = pd.read_csv(test_csv)

valid_df = pd.read_csv(val_csv)
valid_df.head()
import seaborn as sns
sns.distplot(train_df['label'], kde=False)
X_train = train_df.drop('label', axis=1).values

y_train = train_df['label'].values



X_val = valid_df.drop('label', axis=1).values

y_val = valid_df['label'].values
X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_val = X_val.reshape(X_val.shape[0], 28, 28)
X_train = np.expand_dims(X_train, axis=3)

X_val = np.expand_dims(X_val, axis=3)

n_classes = 10 # 0 through 9
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_data_gen = ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
image_data_gen.fit(X_train)
image_data_gen.fit(X_val)
print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)

img_shape = (28, 28, 1)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, UpSampling2D

from tensorflow.keras.optimizers import Adam
model = Sequential()



model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.4))

model.add(BatchNormalization())





model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.4))

model.add(BatchNormalization())



model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.4))

model.add(BatchNormalization())



model.add(UpSampling2D(size=(3, 3), data_format=None, interpolation='nearest'))



model.add(Conv2D(filters=16, kernel_size=(2, 2), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.4))

model.add(BatchNormalization())



model.add(Flatten())





model.add(Dense(128, activation = "relu"))

model.add(Dropout(0.4))



model.add(Dense(10, activation = "softmax"))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit_generator(image_data_gen.flow(X_train, y_train, batch_size=32),

                   epochs=10,

                   validation_data=(image_data_gen.flow(X_val, y_val)))
pd.DataFrame(model.history.history).plot()
test_df.head()
x_test = test_df.drop('id', axis=1).values.reshape(len(test_df), 28, 28, 1)
predictions = model.predict(x_test)
submit_df = pd.DataFrame(columns=['id', 'label'])

submit_df.index.name = 'id'
for index, pred in enumerate(predictions):

    df = pd.DataFrame({

        'id': [index],

        'label': [pred.argmax()]

    })

    submit_df = submit_df.append(df)
submit_df.head()
submit_df.to_csv('./submission.csv', index=False)
model.layers
from tensorflow.keras.models import Model
layers = [layer.output for layer in model.layers]

len(layers)
partial_model = Model(inputs=model.input, outputs=[layers[:19]])
x_test.shape
predicted_partial = partial_model.predict(x_test[220].reshape(1, 28, 28, 1))
import matplotlib.pyplot as plt

plt.imshow(predicted_partial[0][0, :, :, 1])

predicted_partial[0].shape
plt.imshow(x_test[0].reshape(28, 28))

# x_test[0].shape