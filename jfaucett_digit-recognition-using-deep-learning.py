# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# House keeping
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("../input/train.csv")
df.head(10)
import matplotlib.pyplot as plt

train_labels = df['label']
image_data = df.iloc[:, 1:].as_matrix()
print(image_data.shape)
print(train_labels[:5])
print('min: {}, max: {}'.format(np.min(image_data), np.max(image_data)))

# plot the histogram distribution of the labels
plt.hist(train_labels)
plt.ylabel('# of exemplars')
plt.xlabel('ground truth number')
plt.show()

inspection_index = 100

def plot_image(image_data, label):
    plt.title("Number - {}".format(label))
    plt.imshow(image_data.reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.show()
    
plot_image(image_data[6, :], train_labels[6])
from keras.utils import to_categorical

# set the random seed so this is repeatable
np.random.seed(42)
# shuffle our dataset to make sure its good and random
indices = np.arange(len(image_data))
np.random.shuffle(indices)
shuffled_image_data = image_data[indices]
shuffled_label_data = train_labels[indices]

# convert the training labels into one hot encodings
shuffled_labels_one_hot = to_categorical(shuffled_label_data)
print(shuffled_labels_one_hot[:5])

def split_dataset(data, labels, train=0.7, test=0.2, validation=0.1):
    n = len(data)
    train_end_index = round(train * n)
    test_end_index = train_end_index + round(test * n)
    
    train_data = data[:train_end_index]
    train_labels = labels[:train_end_index]
    
    test_data  = data[train_end_index:test_end_index]
    test_labels  = labels[train_end_index:test_end_index]
    
    validation_data = data[test_end_index:]
    validation_labels  = labels[test_end_index:]

    return (train_data, train_labels, test_data, test_labels, validation_data, validation_labels)

x_train, y_train, x_test, y_test, x_val, y_val = split_dataset(shuffled_image_data, shuffled_labels_one_hot)
plot_image(x_train[100], np.argmax(y_train[100]))
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

n_cols = x_train.shape[1]
early_stopping_monitor = EarlyStopping(patience=2)
learning_rate = 0.001

model = Sequential()
model.add(Dense(784, activation='relu', input_shape=(n_cols,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
print(model.summary())

model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
# Fitting the model
epochs = 15
batch_size = 64

hist = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size, callbacks=[early_stopping_monitor])
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

def print_stats(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    print("Accuracy: {:.4f}".format(accuracy_score(y_true, y_pred)))
    print("Precision: {:.4f}".format(precision_score(y_true, y_pred, average='macro')))
    print("Recall: {:.4f}".format(recall_score(y_true, y_pred, average='macro')))
    print('F1: {:.4f}'.format(f1_score(y_true, y_pred, average='macro')))
    
preds = model.predict(x_val)

print_stats(y_val, preds)
from keras.layers import Conv2D, Flatten, Dropout, MaxPooling2D

learning_rate = 0.001

# we need to reshape our data to be (rows, image_width, image_height, image_depth)
x_train2 = x_train.reshape(x_train.shape[0], 28, 28, 1).astype(np.float32)
x_test2 = x_test.reshape(x_test.shape[0], 28, 28, 1).astype(np.float32)
x_val2 = x_val.reshape(x_val.shape[0], 28, 28, 1).astype(np.float32)
x_train2 /= 255
x_test2 /= 255
x_val2 /= 255

plot_image(x_train2[10], np.argmax(y_train[10]))

model = Sequential()
model.add(Conv2D(128, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (5,5), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.summary())

model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
# Fitting the model
epochs = 15
batch_size = 128

hist = model.fit(x_train2, y_train, epochs=epochs, validation_data=(x_test2, y_test), batch_size=batch_size, callbacks=[early_stopping_monitor])
cnn_preds = model.predict(x_val2)

print_stats(y_val, cnn_preds)
# Now Let's try with Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=False)

model = Sequential()
model.add(Conv2D(256, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
print(model.summary())

model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
datagen.fit(x_train2)
epochs = 15

# fits the model on batches with real-time data augmentation:
hist = model.fit_generator(datagen.flow(x_train2, y_train, batch_size=64),
                    steps_per_epoch=len(x_train2) / 32, epochs=epochs, validation_data=(x_test2, y_test), callbacks=[early_stopping_monitor])
## Finally Read in the test.csv and make predictions
df_test = pd.read_csv("../input/test.csv")
df_test.head(2)

sample = pd.read_csv("../input/sample_submission.csv")
sample.head(3)
test_image_data = df_test.as_matrix()
test_data = test_image_data.reshape(test_image_data.shape[0], 28, 28, 1).astype(np.float32)
test_data /= 255

preds_one_hot = model.predict(test_data)
preds = np.argmax(preds_one_hot, axis=1)
image_ids = np.arange(1,len(preds)+1)

out_df = pd.DataFrame({'ImageId' : image_ids, 'Label' : preds })
out_df.head(10)
out_df.to_csv('model_predictions3.csv', index=False)
plot_image(test_data[125], preds[125])
print(len(out_df))