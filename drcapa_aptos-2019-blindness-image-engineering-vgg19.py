import numpy as np

import pandas as pd

import random



import os

path_in = "../input/aptos2019-blindness-detection/"

print(os.listdir(path_in))

print(os.listdir('../input/models'))
import cv2

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer
from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation

from keras.optimizers import RMSprop,Adam

from keras.applications import VGG19
q_size = 150

img_channel = 3

num_classes = 5
train_data = pd.read_csv(path_in+'train.csv')

test_data = pd.read_csv(path_in+'test.csv')

sub_org = pd.read_csv(path_in+'sample_submission.csv')
def plot_bar(data):

    """Simple function to plot the distribution of the classes."""

    dict_data = dict(zip(range(0, num_classes), (((data.value_counts()).sort_index())).tolist()))

    names = list(dict_data.keys())

    values = list(dict_data.values())

    plt.bar(names, values)

    plt.grid()

    plt.show()
def read_images(filepath, data, file_list, size):

    """Read and edit the images of a given folder."""

    for file in file_list:

        img = cv2.imread(filepath+file+'.png')

        img = cv2.resize(img, (size, size))

        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)

        data[file_list.index(file), :, :, :] = img
def add_flip_image(data, file_list):

    """Simple function to flip images by a given list."""

    temp = np.empty((1, data.shape[1], data.shape[2], data.shape[3]), dtype=np.uint8)

    for index in file_list.index:

        img = data[index, :, :, :]

        vertical_img = cv2.flip(img, 1)

        temp[0, :, :, :] = vertical_img

        data = np.concatenate((data, temp), axis=0)

    return data
def add_rot_image(data, file_list):

    """Simple function to rotate images by a given list."""

    degrees = 15

    temp = np.empty((1, data.shape[1], data.shape[2], data.shape[3]), dtype=np.uint8)

    for index in file_list.index:

        img = data[index, :, :, :]

        rows,cols, channel = img.shape

        Matrix = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)

        rotate_img = cv2.warpAffine(img, Matrix, (cols, rows))

        temp[0, :, :, :] = rotate_img

        data = np.concatenate((data, temp), axis=0)

    return data
def add_zoom_image(data, file_list):

    """Simple function to zoom in images by a given list."""

    temp = np.empty((1, data.shape[1], data.shape[2], data.shape[3]), dtype=np.uint8)

    for index in file_list.index:

        img = data[index, :, :, :]

        size = img.shape[0]

        pts1 = np.float32([[10,10],[size-10, 10],[10, size-10],[size-10, size-10]])

        pts2 = np.float32([[0, 0],[size-20, 0],[0, size-20],[size-20, size-20]])

        Matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # zoom image

        img_zoom = cv2.warpPerspective(img, Matrix, (size-20, size-20))

        dim = img.shape 

        # resize image

        rows,cols, channel = img.shape

        dim=(rows, cols)

        img_scale = cv2.resize(img_zoom, dim, interpolation = cv2.INTER_AREA)

        temp[0, :, :, :] = img_scale

        data = np.concatenate((data, temp), axis=0)

    return data
def get_multilabel(diagnosis):

    """A function to get multi-label from single-label."""

    return ','.join([str(i) for i in range(diagnosis + 1)])
X_train_org = np.empty((len(train_data), q_size, q_size, img_channel), dtype=np.uint8)

X_test = np.empty((len(test_data), q_size, q_size, img_channel), dtype=np.uint8)
read_images(path_in+'train_images/', X_train_org, train_data['id_code'].tolist(), q_size)

read_images(path_in+'test_images/', X_test, sub_org['id_code'].tolist(), q_size)
plot_bar(train_data['diagnosis'])
list_flip = train_data[train_data['diagnosis'] != 0]
X_train_org = add_flip_image(X_train_org, list_flip)

train_data = train_data.append(list_flip, ignore_index=True, sort=False)
plot_bar(train_data['diagnosis'])
list_rot = train_data[(train_data['diagnosis'] != 0)&

                      (train_data['diagnosis'] != 2)]
X_train_org = add_rot_image(X_train_org, list_rot)

train_data = train_data.append(list_rot, ignore_index=True, sort=False)
plot_bar(train_data['diagnosis'])
list_zoom = train_data[(train_data['diagnosis'] == 3)|

                       (train_data['diagnosis'] == 4)]
X_train_org = add_zoom_image(X_train_org, list_zoom)

train_data = train_data.append(list_zoom, ignore_index=True, sort=False)
plot_bar(train_data['diagnosis'])
num_val = (train_data['diagnosis'].value_counts()).min()

list_new = []

for i in range(num_classes):

    temp = random.choices(train_data[train_data['diagnosis']==i].index, k=num_val)

    list_new.extend(temp)

train_data = train_data.loc[list_new]

X_train_org = X_train_org[list_new]
plot_bar(train_data['diagnosis'])
image_number=4019

print(train_data.iloc[image_number])

plt.imshow(X_train_org[image_number], cmap='gray')

plt.show()
train_data['multilabel'] = train_data['diagnosis'].apply(get_multilabel)
category =['0','1','2','3','4']

MLB = MultiLabelBinarizer(category)

y_train_org_multi = MLB.fit_transform(train_data['multilabel']).astype('float32')
class_weight = dict(zip(range(0, num_classes), (((train_data['diagnosis'].value_counts()).sort_index())/len(train_data)).tolist()))
mean = X_train_org.mean(axis=0)

X_train_org = X_train_org.astype('float32')

X_train_org -= X_train_org.mean(axis=0)

std = X_train_org.std(axis=0)

X_train_org /= X_train_org.std(axis=0)

X_test = X_test.astype('float32')

X_test -= mean

X_test /= std
X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_org_multi,

                                                  test_size=0.1, random_state=0)
conv_base = VGG19(weights='../input/models/model_weights_vgg19.h5',

                  include_top=False,

                  input_shape=(q_size, q_size, img_channel))

conv_base.trainable = True
model = Sequential()

model.add(conv_base)

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(5, activation='sigmoid'))
model.compile(optimizer = Adam(lr=5e-7),

              loss='binary_crossentropy',

              metrics=['binary_accuracy'])
model.summary()
epochs = 100

batch_size = 32
history = model.fit(X_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    validation_data=(X_val, y_val),

                    class_weight=class_weight)
y_test = model.predict(X_test)
y_test_classes = np.where(y_test>0.5, 1, 0).sum(axis=1)-1
output = pd.DataFrame({'id_code': sub_org['id_code'],

                       'diagnosis': y_test_classes})

output.to_csv('submission.csv', index=False)
plot_bar(output['diagnosis'])
loss = history.history['loss']

loss_val = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='loss_train')

plt.plot(epochs, loss_val, 'b', label='los_val')

plt.title('Value of the loss-function')

plt.xlabel('Epochs')

plt.ylabel('Value of the loss-function')

plt.legend()

plt.grid()

plt.show()
acc = history.history['binary_accuracy']

acc_val = history.history['val_binary_accuracy']

epochs = range(1, len(loss)+1)

plt.plot(epochs, acc, 'bo', label='Accuracy_Train')

plt.plot(epochs, acc_val, 'b', label='Accuracy_Val')

plt.title('Value of the accurarcy')

plt.xlabel('Epochs')

plt.ylabel('Value of the accuracy')

plt.legend()

plt.grid()

plt.show()
del model