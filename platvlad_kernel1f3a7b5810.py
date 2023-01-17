# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



dir_name = '/kaggle/input/tl-signs-hse-itmo-2020-winter'

labels_data = pd.read_csv(os.path.join(dir_name, 'train.csv'))

labels = labels_data['class_number'].to_numpy()



def get_img_data(path):

    full_path = os.path.join(dir_name, path)

    img_files = sorted(os.listdir(full_path))

    data = None

    for i in range(len(img_files)):

        file_name = img_files[i]

        file_path = os.path.join(full_path, file_name)

        img = cv2.imread(file_path)

        if data is None:

            data = np.zeros(shape=(len(img_files), img.shape[0], img.shape[1], img.shape[2]))

        data[i] = img

    

    return data / 255.0



img_data = get_img_data('train/train')

import keras

import tensorflow as tf





model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(48, (3, 3), activation=tf.nn.relu, input_shape=(48, 48, 3)),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation=tf.nn.relu),

    tf.keras.layers.Dense(67, activation=tf.nn.softmax)

])

model.compile(optimizer=tf.keras.optimizers.Adam(), 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.fit(img_data, labels, epochs=8)
test_data = get_img_data('test/test')



predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
answer_column = pd.DataFrame(predicted_classes)

submission_frame = pd.read_csv(os.path.join(dir_name, 'sample_submission.csv'))

submission_frame['class_number'] = answer_column

submission_frame.to_csv('answer_neural.csv', index=False)