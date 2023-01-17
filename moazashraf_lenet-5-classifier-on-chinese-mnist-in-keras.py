import os.path as osp
import glob
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# load the data
IMAGE_DIR="../input/chinese-mnist/data/data"


def load_data():
    file_list = glob.glob(IMAGE_DIR + "/*.jpg")
    x = []
    y = []

    for fname in file_list:
        with Image.open(fname) as img:
            np_img = np.array(img)
        label = int(osp.split(fname)[-1].split('.')[0].split('_')[3])-1   # totally unreadable, unclean code

        x.append(np_img)
        y.append(label)
    x, y = np.array(x), np.array(y)
    x = np.expand_dims(x, -1)
    x = x / 255.
    return x, y
    
x, y = load_data()
# split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

# create the model
model = Sequential([
    Conv2D(6, 5, activation='relu', input_shape=x_train.shape[1:]),
    MaxPool2D(2),
    Conv2D(16, 5, activation='relu',),
    MaxPool2D(2),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(15, activation='softmax')
])
model.summary()
from tensorflow.keras.optimizers import Adam

# configure the model for training
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['acc'])
# train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=4, verbose=2)
code_to_kanji = ['零', '一', '二', '三', '四', '五', '六', '七',
                 '八', '九', '十', '百', '千', '万', '亿']
# show an example
example_idx = 10
x_example = x_test[example_idx]
y_example = y_test[example_idx]
y_pred_ex = model(np.expand_dims(x_example, 0))
y_pred_ex = np.argmax(y_pred_ex)
print(code_to_kanji[y_pred_ex])
plt.imshow(x_example.squeeze(2), cmap='gray')
plt.axis(False)
plt.show()
# show a wrong prediction
y_pred_test = model(x_test)
y_pred_test = np.argmax(y_pred_test, -1)
error_mask = (y_pred_test != y_test)

x_error = x_test[error_mask]
y_pred_error = y_pred_test[error_mask]
y_error = y_test[error_mask]

idx = 10
print(f"prediction: {code_to_kanji[y_pred_error[idx]]}, truth: {code_to_kanji[y_error[idx]]}")
plt.imshow(x_error[idx].squeeze(2), cmap='gray')
plt.axis(False)
plt.show()