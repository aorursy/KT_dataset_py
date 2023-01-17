import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf

import os



batch_size = 40

%matplotlib inline
path = '../input/train/train/'
NSFW = os.listdir(path + 'NSFW/')

SFW = os.listdir(path + 'SFW/')
resnet = tf.keras.applications.resnet50.ResNet50(include_top=False)

preprocess_input = tf.keras.applications.resnet50.preprocess_input

image = tf.keras.preprocessing.image
def extract_features(img_paths, batch_size=batch_size):

    """ This function extracts image features for each image in img_paths using ResNet50 bottleneck layer.

        Returned features is a numpy array with shape (len(img_paths), 2048).

    """

    global resnet

    n = len(img_paths)

    img_array = np.zeros((n, 299, 299, 3))

    

    for i, path in enumerate(img_paths):

        img = image.load_img(path, target_size=(299, 299))

        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)

        x = preprocess_input(img)

        img_array[i] = x

    

    X = resnet.predict(img_array, batch_size=batch_size, verbose=1)

    X = X.reshape(n, 2048, -1)

    return X

X = extract_features(

    list(map(lambda x: path + 'NSFW/' + x, NSFW)) + list(map(lambda x: path + 'SFW/' + x, SFW))

)

y = np.array([1] * len(NSFW) + [0] * len(SFW))
def net():

    model = tf.keras.models.Sequential([

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(256, activation=tf.nn.relu),

      tf.keras.layers.Dropout(0.6),

      tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    ])

    return model
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
model = net()
X_train.shape
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Dense



np.random.seed(42)



epochs = 10



model = net()

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



history = model.fit(X_train, y_train,

                    validation_data=(X_val, y_val),

                    batch_size=batch_size,

                    epochs=epochs)
plt.plot(range(1,epochs+1), history.history['acc'], label='train')

plt.plot(range(1,epochs+1), history.history['val_acc'], label='validation')

plt.legend()
plt.plot(range(1,epochs+1), history.history['loss'], label='train loss')

plt.plot(range(1,epochs+1), history.history['val_loss'], label='val loss')

plt.legend()
model.summary()
test_path = '../input/test/test/'

test = os.listdir(test_path)
X_test = extract_features(

    list(map(lambda x: test_path + x, test))

)
y_pred = model.predict(X_test)
pred = pd.DataFrame({

    'id': test,

    'kelas': (y_pred > .5).reshape(-1)

})

pred['kelas'] = pred['kelas'].map({True: 1, False: 0})
resnet.summary()
pred.to_csv('pred_1.csv', index=False)
pred