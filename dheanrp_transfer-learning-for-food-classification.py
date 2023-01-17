import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf

import os



batch_size = 32

%matplotlib inline
path = '../input/train/train/'
fried_rice = os.listdir(path + 'fried_rice/')

ramen = os.listdir(path + 'ramen/')
resnet = tf.keras.applications.xception.Xception(include_top=False)

preprocess_input = tf.keras.applications.xception.preprocess_input

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

    list(map(lambda x: path + 'fried_rice/' + x, fried_rice)) + list(map(lambda x: path + 'ramen/' + x, ramen))

)

y = np.array([1] * len(fried_rice) + [0] * len(ramen))
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
def net():

    model = tf.keras.models.Sequential([

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(256, activation=tf.nn.relu),

      tf.keras.layers.Dropout(0.2),

      tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    ])

    return model
model = net()
X_train.shape
from keras.models import Sequential

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense



np.random.seed(42)



epochs = 5



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
pred.to_csv('pred_1.csv', index=False)