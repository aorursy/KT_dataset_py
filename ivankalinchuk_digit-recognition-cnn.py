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
digits_csv = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_digits_csv = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
digit_column_names = list(digits_csv.columns)
print('digit csv COLUMNS:\n',digits_csv.columns)
print('there are', len(digit_column_names),'columns in the train csv and',len(digits_csv),'lines')
image_size = (28, 28)
image_target_shape = image_size + (1,)
from random import shuffle
valid_part = 0.15
divide_index = int(len(digits_csv) * valid_part)
is_train_line = [False for _ in range(divide_index)] + [True for _ in range(divide_index, len(digits_csv))]
shuffle(is_train_line)
assert len(is_train_line) == len(digits_csv)
pixel_headers = digit_column_names[1:] 
def get_pixels_from_digit_csv(row_id, data_csv):
    pixels = []
    for pix_col in pixel_headers:
        pixels.append(data_csv[pix_col][row_id])
    return np.array(pixels)
x_train = digits_csv.drop(labels = ['label'], axis = 1) / 255.
y_train = digits_csv['label']
x_train = [val.reshape(28, 28, 1) for val in np.array(x_train)]
y_train = tf.keras.utils.to_categorical(np.array(y_train))
print('there are',len(x_train),'images for test (will be seperated for valid)')
import tensorflow as tf
num_of_classes = len(np.unique(y_train))
print('number of classes=', num_of_classes)
import matplotlib.pyplot as plt
random_images = x_train
np.random.shuffle(random_images)
popularity_map = np.zeros(image_size)
for rand_img in random_images:
    popularity_map += rand_img
plt.grid(True)    
plt.imshow(popularity_map, label='popularity_map')
plt.show()
digit_classification_cnn_model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                                                                    input_shape=(28, 28, 1)),
                                                             tf.keras.layers.MaxPooling2D(2, 2),
                                                             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                                             tf.keras.layers.MaxPooling2D(2, 2),
                                                             tf.keras.layers.Dropout(0.5),
                                                             tf.keras.layers.Flatten(),
                                                             tf.keras.layers.Dense(64, activation='relu'),
                                                             tf.keras.layers.Dense(num_of_classes, activation='softmax')
                                                                ])

digit_classification_cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])

digit_classification_cnn_model.summary()

x_train = np.array(x_train)
y_train = np.array(y_train)
fit_history = digit_classification_cnn_model.fit(x_train, y_train, validation_split=0.25,
                                                epochs=50, verbose=1)
import matplotlib.pyplot as plt
plt.plot(range(len(fit_history.history['accuracy'])), fit_history.history['accuracy'])
plt.plot(range(len(fit_history.history['val_accuracy'])), fit_history.history['val_accuracy'], '--')
plt.grid(True)
plt.show()
test_images = np.array(test_digits_csv / .255).reshape(len(test_digits_csv), 28, 28, 1)
print('there are', len(test_images),'images for test')
print('images for test shape', test_images.shape[0])
predictions = digit_classification_cnn_model.predict(test_images)
print(predictions)
def get_class_label(prediction):
    max_val = -1
    id_max = 0
    for i, possibility in enumerate(prediction):
        if possibility > max_val:
            max_val = possibility
            id_max = i
    return id_max
prediction_classes = [get_class_label(pred) for pred in predictions]
#zero - 9
#one - 
print(prediction_classes)
my_prediction_dict = {'ImageId' : range(1, len(prediction_classes) + 1),
                 'Label' : prediction_classes}

my_prediction_dataframe = pd.DataFrame(my_prediction_dict)
my_prediction_dataframe.to_csv('mydigitprediction.csv',index=False)
num_of_examples = 15
example_figure = plt.figure(figsize=(10, 10))
for i in range(num_of_examples):
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_images[i].reshape(28,28))
    plt.title('prediction={}'.format(prediction_classes[i]))
plt.tight_layout()
my_prediction_dataframe.head()