import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from IPython.display import Image

Image("../input/sign-language-mnist/amer_sign2.png")
train_data = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')

test_data = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')
train_data.head()
train_label = train_data['label']

train_data.drop('label', axis=1, inplace=True)
train_data.head()
def image_show(data_row):

    img = np.array(data_row).reshape(28,28)

    plt.imshow(img)
image_show(train_data.iloc[0])
from keras.utils import to_categorical

train_label = to_categorical(train_label)

train_label[0]
image = train_data.values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(image, train_label, test_size=0.3, random_state=42)
x_train = x_train.reshape(x_train.shape[0],28,28,1)

x_test = x_test.reshape(x_test.shape[0],28,28,1)
image_show(x_train[0])
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
model = Sequential([

    Conv2D(64,(3,3),activation='relu', input_shape=(28,28,1)),

    MaxPooling2D(2,2),

    Dropout(0.2),

    Conv2D(64,(3,3),activation='relu'),

    MaxPooling2D(2,2),

    Dropout(0.2),

    Conv2D(64,(3,3),activation='relu'),

    MaxPooling2D(2,2),

    Dropout(0.2),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.2),

    Dense(64,activation='relu'),

    Dense(25, activation='softmax')

])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_class=0

batch=128

epoch=50
history = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=epoch, batch_size=batch)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])
test_labels = test_data['label']

test_data.drop('label', axis=1,inplace= True)
test_labels = to_categorical(test_labels)

test_image = test_data.values

test_image = test_image.reshape(test_image.shape[0],28,28,1)
y_pred = model.predict(test_image)
from sklearn.metrics import accuracy_score
accuracy_score(test_labels, y_pred.round())