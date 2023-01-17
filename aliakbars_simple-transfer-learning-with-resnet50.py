# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from glob import glob

print(glob("../input/*/*/*/"))



# Any results you write to the current directory are saved as output.
resnet = tf.keras.applications.resnet50.ResNet50(include_top=False)

preprocess_input = tf.keras.applications.resnet50.preprocess_input

image = tf.keras.preprocessing.image



def extract_features(img_paths):

    n = len(img_paths)

    img_array = np.zeros((n, 224, 224, 3))

    

    for i, path in enumerate(img_paths):

        img = image.load_img(path, target_size=(224, 224))

        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)

        x = preprocess_input(img)

        img_array[i] = x

    

    return img_array



def create_model(base_model, fine_tune_all=False):

    base_model.trainable = fine_tune_all # freeze base model layers



    model = tf.keras.Sequential([

        base_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(100, activation='relu', name='fc'), # add a fully connected layer

        tf.keras.layers.Dense(1, activation='sigmoid', name='output')

    ])

    

    return model
train = glob('../input/*/*/*/*')
X_train = extract_features(train)

y_train = list(map(lambda x: x.split('/')[-2], train))

y_train = np.array(y_train) == 'fried_rice'
batch_size = 128

epochs = 16



model = create_model(resnet)

model.compile(

    optimizer=tf.keras.optimizers.Adam(),

    loss='binary_crossentropy',

    metrics=['accuracy']

)



history = model.fit(

    X_train, y_train,

    batch_size=batch_size,

    epochs=epochs,

    validation_split=0.3

)
plt.plot(range(1,17), history.history['acc'], label='train')

plt.plot(range(1,17), history.history['val_acc'], label='val')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend();
test = glob('../input/test/*/*')

X_test = extract_features(test)

y_pred = model.predict(X_test)