import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from skimage import io, transform
import matplotlib.pyplot as plt
import random
import os
IMAGE_WIDTH=200
IMAGE_HEIGHT=200
IMAGE_CHANNELS=3
EPOCHS=30
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
PATH='../input/dataset-for-mask-detection/dataset/'
PATH2= '../input/face-mask-detection/dataset/'
# 데이터 불러오기
with_mask = os.listdir(PATH+"with_mask")
without_mask = os.listdir(PATH+"without_mask")
with_mask2 = os.listdir(PATH2+"with_mask")
without_mask2 = os.listdir(PATH2+"without_mask")


def add_path1(filename):
    return PATH +'with_mask/' + filename
def add_path2(filename):
    return PATH + 'without_mask/' + filename
def add_path3(filename):
    return PATH2 +'with_mask/' + filename
def add_path4(filename):
    return PATH2 + 'without_mask/' + filename

w_mask = list(map(add_path1, with_mask))
wo_mask = list(map(add_path2, without_mask))
w_mask2 = list(map(add_path3, with_mask2))
wo_mask2 = list(map(add_path4, without_mask2))


# 데이터 preprocessing & label

def dataset(file_list_with, file_list_without,file_list_with2, file_list_without2,size=IMAGE_SIZE,flattened=False):
    data = []
    labels = []
    sum_1 = 0
    sum_2 = 0
    for i, file in enumerate(file_list_with):
        if(file == PATH + "with_mask/.ipynb_checkpoints"):
            continue
        image = io.imread(file)
        image = transform.resize(image, size, mode='constant')
        data.append(image)
        labels.append(1)
    for i, file in enumerate(file_list_without):
        if(file == PATH + "without_mask/.ipynb_checkpoints"):
            continue
        image = io.imread(file)
        image = transform.resize(image, size, mode='constant')
        data.append(image)
        labels.append(0)
    for i, file in enumerate(file_list_with2):
        if(file == PATH2 + "with_mask/.ipynb_checkpoints"):
            continue
        image = io.imread(file)
        image = transform.resize(image, size, mode='constant')
        if(image.shape == (200,200,4)):
            sum_1 += 1
            continue
        data.append(image)
        labels.append(1)
    for i, file in enumerate(file_list_without2):
        if(file == PATH2 + "without_mask/.ipynb_checkpoints"):
            continue
        image = io.imread(file)
        image = transform.resize(image, size, mode='constant')
        if(image.shape == (200,200,4)):
            sum_2 += 1
            continue
        data.append(image)
        labels.append(0)
    
    print(sum_1, sum_2)
    return np.array(data), np.array(labels)
# skimage 의 transform.resize 가 auto scale 되서 나오는듯합니다.
# 0-1 의 범위를 가지고 있습니다.

X, y = dataset(w_mask, wo_mask,w_mask2, wo_mask2)
print(X.shape,y.shape)


# 데이터 확인하기
sample_1 = random.choice(X)

f = plt.figure()
plt.imshow(sample_1)
plt.show(block=True)


# create model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, BatchNormalization, MaxPooling2D, Dropout
def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', strides=(2,2), input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model
model1 = create_model()
model1.summary()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10,stratify=y)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
partial_x_train, validation_x_train, partial_y_train, validation_y_train = train_test_split(x_train, y_train, test_size=0.20)
print(partial_x_train.shape,validation_x_train.shape,partial_y_train.shape,validation_y_train.shape)
print('The size of the training set: ',len(x_train))
print('The size of the partial training set: ',len(partial_x_train))
print('The size of the validation training set: ',len(validation_x_train))
print('The size of the testing set: ',len(x_test))
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [learning_rate_reduction]
history = model1.fit(
    partial_x_train, 
    partial_y_train,
    validation_data=(validation_x_train, validation_y_train),
    epochs=EPOCHS, 
    batch_size=32,
    verbose =1,
    callbacks=callbacks
)

def smooth_curve(points, factor=0.8): #this function will make our plots more smooth
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
plt.plot(epochs, smooth_curve(val_acc), 'r-', label='Validation acc')
plt.legend()
plt.title('Training and Validation Acc')
plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'r-', label='Validation loss')
plt.legend()
plt.title('Training and Validation loss')
plt.show()
test_loss, test_acc = model1.evaluate(x_test, y_test, steps=32)
print('The final test accuracy: ',test_acc)
predictions = model1.predict(x_test)     # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability
def print_mislabeled_images(test_images, test_labels, pred_labels):
    """
        Print 25 examples of mislabeled images by the classifier, e.g when test_labels != pred_labels
    """
    BOO = (test_labels == pred_labels)
    mislabeled_indices = random.choice(np.where(BOO == 0)[0])
    mislabeled_images = test_images[mislabeled_indices]
    mislabeled_labels = pred_labels[mislabeled_indices]
    print(mislabeled_labels)
    title = "Some examples of mislabeled images by the classifier:"
    f = plt.figure()
    plt.imshow(mislabeled_images)
    plt.show(block=True)

print_mislabeled_images(x_test, y_test, pred_labels)
