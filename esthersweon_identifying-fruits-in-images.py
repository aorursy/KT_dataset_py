import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import skimage
from skimage import io, transform
from IPython.display import Image, display

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import LSTM, Input
from keras.models import Model
from keras.optimizers import Adam

img_size = 100
train_dir = './data/fruits/train/'
test_dir = './data/fruits/test/'

def get_data(folder_path):
    imgs = []
    indices = []
    labels = []
    for idx, folder_name in enumerate(os.listdir(folder_path)[:10]):
        if not folder_name.startswith('.'):
            labels.append(folder_name)
            for file_name in tqdm(os.listdir(folder_path + folder_name)):
                if not file_name.startswith('.'):
                    img_file = io.imread(folder_path + folder_name + '/' + file_name)
                    if img_file is not None:
                        img_file = transform.resize(img_file, (img_size, img_size))
                        imgs.append(np.asarray(img_file))
                        indices.append(idx)
    imgs = np.asarray(imgs)
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    return imgs, indices, labels

X_train, y_train, train_labels = get_data(train_dir)
X_test, y_test, test_labels = get_data(test_dir)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train:', y_train)
print('y_test:', y_test)
# print('First image - X_train:', X_train[0])

num_categories = len(np.unique(y_train))

new_X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype('float32')
new_X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype('float32')
new_y_train = keras.utils.to_categorical(y_train, num_categories)
new_y_test = keras.utils.to_categorical(y_test, num_categories)
def display_imgs(folder_path):
    for idx, folder_name in enumerate(os.listdir(folder_path)):
        if idx % 25 == 0:
            if not folder_name.startswith('.'):
                for idx2, file_name in enumerate(tqdm(os.listdir(folder_path + folder_name))):
                    if idx2 % 75 == 0:
                        if not file_name.startswith('.'):
                            img_filename = folder_path + folder_name + '/' + file_name
                            display(Image(filename=img_filename))
display_imgs(train_dir)
display_imgs(test_dir)
def evaluate_model(model, batch_size, epochs):
    history = model.fit(new_X_train, new_y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(new_X_test, new_y_test))
    score = model.evaluate(new_X_test, new_y_test, verbose=0)
    print('***Metrics Names***', model.metrics_names)
    print('***Metrics Values***', score)
convolutional = Sequential()

convolutional.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
convolutional.add(Conv2D(64, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.25))

convolutional.add(Flatten())
convolutional.add(Dense(128, activation='relu'))
convolutional.add(Dropout(0.5))
convolutional.add(Dense(num_categories, activation='softmax'))

convolutional.summary()
convolutional.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

evaluate_model(convolutional, 128, 5)
convolutional = Sequential()

convolutional.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
convolutional.add(Conv2D(64, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.6))

convolutional.add(Flatten())
convolutional.add(Dense(128, activation='relu'))
convolutional.add(Dropout(0.6))
convolutional.add(Dense(num_categories, activation='softmax'))

convolutional.summary()
convolutional.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

evaluate_model(convolutional, 128, 7)
def run_with_loss(loss):
    convolutional = Sequential()

    convolutional.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
    convolutional.add(Conv2D(64, (3, 3), activation='relu'))
    convolutional.add(MaxPooling2D(pool_size=(2, 2)))
    convolutional.add(Dropout(0.6))

    convolutional.add(Flatten())
    convolutional.add(Dense(128, activation='relu'))
    convolutional.add(Dropout(0.6))
    convolutional.add(Dense(num_categories, activation='softmax'))

    convolutional.summary()
    convolutional.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

    evaluate_model(convolutional, 128, 5)

losses = ['mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error']

for loss in losses:
    run_with_loss(loss)
convolutional = Sequential()

convolutional.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
convolutional.add(Conv2D(64, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

# CHANGE
convolutional.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
convolutional.add(Conv2D(256, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

convolutional.add(Flatten())
convolutional.add(Dense(512, activation='relu'))
convolutional.add(Dropout(0.6))
convolutional.add(Dense(num_categories, activation='softmax'))

convolutional.summary()
convolutional.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

evaluate_model(convolutional, 128, 5)
convolutional = Sequential()

convolutional.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
convolutional.add(Conv2D(64, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

# CHANGE
convolutional.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
convolutional.add(Conv2D(256, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

convolutional.add(Flatten())
convolutional.add(Dense(512, activation='relu'))
convolutional.add(Dropout(0.6))
convolutional.add(Dense(num_categories, activation='softmax'))

convolutional.summary()
convolutional.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

evaluate_model(convolutional, 32, 5)
convolutional = Sequential()

convolutional.add(Conv2D(32, kernel_size=(2,2), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
convolutional.add(Conv2D(64, (2, 2), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

# CHANGE
convolutional.add(Conv2D(128, kernel_size=(2,2), activation='relu'))
convolutional.add(Conv2D(256, (2, 2), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

convolutional.add(Flatten())
convolutional.add(Dense(512, activation='relu'))
convolutional.add(Dropout(0.6))
convolutional.add(Dense(num_categories, activation='softmax'))

convolutional.summary()
convolutional.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

evaluate_model(convolutional, 32, 5)
convolutional = Sequential()

convolutional.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
convolutional.add(Conv2D(64, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

# CHANGE
convolutional.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
convolutional.add(Conv2D(256, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

convolutional.add(Flatten())
convolutional.add(Dense(512, activation='relu'))
convolutional.add(Dropout(0.6))
convolutional.add(Dense(num_categories, activation='softmax'))

convolutional.summary()
convolutional.compile(loss="categorical_crossentropy", optimizer=Adam(lr=.0005), metrics=['accuracy'])

evaluate_model(convolutional, 32, 5)
convolutional = Sequential()

convolutional.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
convolutional.add(Conv2D(64, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

# CHANGE
convolutional.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
convolutional.add(Conv2D(256, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

convolutional.add(Flatten())
convolutional.add(Dense(512, activation='relu'))
convolutional.add(Dropout(0.6))
convolutional.add(BatchNormalization())
convolutional.add(Dense(num_categories, activation='softmax'))

convolutional.summary()
convolutional.compile(loss="categorical_crossentropy", optimizer=Adam(lr=.0005), metrics=['accuracy'])

evaluate_model(convolutional, 32, 5)
y_pred = convolutional.predict(new_X_test, batch_size=None, verbose=0, steps=None).argmax(axis=-1)
res_crosstab = pd.crosstab(y_pred, y_test)

dict_idx_fruit = {idx: label for idx, label in enumerate(test_labels)}
print(dict_idx_fruit)

res_crosstab
for idx in range(num_categories):
    accuracy = res_crosstab.loc[idx, idx] / res_crosstab.loc[:, idx].sum()
    flag = '***LOW***' if accuracy < 0.8 else ''
    print(dict_idx_fruit[idx])
    print('   ', flag, 'accuracy –', round(accuracy * 100, 2), '%')
# Run on Kaggle with GPU 

import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import skimage
from skimage import io, transform
from IPython.display import Image, display

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import LSTM, Input
from keras.models import Model
from keras.optimizers import Adam

img_size = 100
train_dir = '../input/fruits-360_dataset/fruits-360/Training/'
test_dir = '../input/fruits-360_dataset/fruits-360/Test/'

def get_data(folder_path):
    imgs = []
    indices = []
    labels = []
    for idx, folder_name in enumerate(os.listdir(folder_path)[:35]):
        if not folder_name.startswith('.'):
            labels.append(folder_name)
            for file_name in tqdm(os.listdir(folder_path + folder_name)):
                if not file_name.startswith('.'):
                    img_file = io.imread(folder_path + folder_name + '/' + file_name)
                    if img_file is not None:
                        img_file = transform.resize(img_file, (img_size, img_size))
                        imgs.append(np.asarray(img_file))
                        indices.append(idx)
    imgs = np.asarray(imgs)
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    return imgs, indices, labels

X_train, y_train, train_labels = get_data(train_dir)
X_test, y_test, test_labels = get_data(test_dir)
num_categories = len(np.unique(y_train))

new_X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype('float32')
new_X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype('float32')
new_y_train = keras.utils.to_categorical(y_train, num_categories)
new_y_test = keras.utils.to_categorical(y_test, num_categories)
def evaluate_model(model, batch_size, epochs):
    history = model.fit(new_X_train, new_y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(new_X_test, new_y_test))
    score = model.evaluate(new_X_test, new_y_test, verbose=0)
    print('***Metrics Names***', model.metrics_names)
    print('***Metrics Values***', score)
convolutional = Sequential()

convolutional.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
convolutional.add(Conv2D(64, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

convolutional.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
convolutional.add(Conv2D(256, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

convolutional.add(Flatten())
convolutional.add(Dense(512, activation='relu'))
convolutional.add(Dropout(0.6))
convolutional.add(BatchNormalization())
convolutional.add(Dense(num_categories, activation='softmax'))

convolutional.summary()
convolutional.compile(loss="categorical_crossentropy", optimizer=Adam(lr=.0005), metrics=['accuracy'])

evaluate_model(convolutional, 32, 5)
# Save model to disk
model_json = convolutional.to_json()
with open("cnn_model.json", "w") as json_file:
    json_file.write(model_json)
convolutional.save_weights("cnn_model.h5")
print("CNN model saved to disk")
# Load model from disk
json_file = open('cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("cnn_model.h5")
print("CNN model loaded from disk")
y_pred = loaded_model.predict(new_X_test, batch_size=None, verbose=0, steps=None).argmax(axis=-1)
res_crosstab = pd.crosstab(y_pred, y_test)

dict_idx_fruit = {idx: label for idx, label in enumerate(test_labels)}
print(dict_idx_fruit)

res_crosstab
for idx in range(num_categories):
    accuracy = res_crosstab.loc[idx, idx] / res_crosstab.loc[:, idx].sum()
    flag = '***LOW***' if accuracy < 0.75 else ''
    print(dict_idx_fruit[idx])
    print('   ', flag, 'accuracy –', round(accuracy * 100, 2), '%')
for idx in range(35):
    for idx2 in range(35):
        accuracy = res_crosstab.loc[idx, idx] / res_crosstab.loc[:, idx].sum()
        if idx != idx2 and res_crosstab.loc[idx, idx2] != 0:
            pred_fruit = dict_idx_fruit[idx]
            actual_fruit = dict_idx_fruit[idx2]
            num_mistakes = res_crosstab.loc[idx, idx2]
            sing_or_plural = 's' if num_mistakes > 1 else ''
            print('- {} {}{} mistaken for {}{}'.format(num_mistakes, actual_fruit, sing_or_plural, pred_fruit, sing_or_plural))
def get_one_img_per_fruit(folder_path):
    printouts = []
    for idx, folder_name in enumerate(os.listdir(folder_path)[:35]):
        if not folder_name.startswith('.'):
            for idx2, file_name in enumerate(tqdm(os.listdir(folder_path + folder_name))):
                if idx2 == 0:
                    if not file_name.startswith('.'):
                        img_filename = folder_path + folder_name + '/' + file_name
                        display(Image(filename=img_filename))
                        
                        current_img = io.imread(img_filename)
                        current_img = transform.resize(current_img, (img_size, img_size))
                        current_img = np.asarray(current_img)
                        current_img = np.asarray([current_img])
                        
                        current_pred = loaded_model.predict(current_img, batch_size=None, verbose=0, steps=None).argmax(axis=-1)
                        current_pred = dict_idx_fruit[current_pred[0]]
                        
                        is_incorrect = 'INCORRECT' if folder_name != current_pred else ''
                        
                        msg = '{} – predicted as {} {}'.format(folder_name, current_pred, is_incorrect)
                        print(msg)
                        printouts.append(msg)
    return printouts
                    
printouts = get_one_img_per_fruit(test_dir)

for msg in printouts:
    print(msg)
