import numpy as np
import pandas as pd
import seaborn as sns
import os
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
train_data = []
img_size = 64
assign_dict = {"cats":0, "dogs":1}
directory = "../input/cat-and-dog/training_set/training_set"
for sub_directory in os.listdir(directory):
    if sub_directory == "cats":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            try:
                img = cv2.imread(os.path.join(inner_directory,i),1)
                img = cv2.resize(img,(img_size,img_size))
                train_data.append([img,assign_dict[sub_directory]])
            except:
                pass
    if sub_directory == "dogs":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            try:
                img = cv2.imread(os.path.join(inner_directory,i),1)
                img = cv2.resize(img,(img_size,img_size))
                train_data.append([img,assign_dict[sub_directory]])
            except:
                pass
random.shuffle(train_data)
len(train_data)
sns.set_style('darkgrid')
p = []
for animal in train_data:
    if(animal[1] == 0):
        p.append("Cat")
    else:
        p.append("Dog")
sns.countplot(p)
test_data = []
img_size = 64
assign_dict = {"cats":0, "dogs":1}
directory = "../input/cat-and-dog/test_set/test_set"
for sub_directory in os.listdir(directory):
    if sub_directory == "cats":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            try:
                img = cv2.imread(os.path.join(inner_directory,i),1)
                img = cv2.resize(img,(img_size,img_size))
                test_data.append([img,assign_dict[sub_directory]])
            except:
                pass
    if sub_directory == "dogs":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            try:
                img = cv2.imread(os.path.join(inner_directory,i),1)
                img = cv2.resize(img,(img_size,img_size))
                test_data.append([img,assign_dict[sub_directory]])
            except:
                pass
len(test_data)
train_X = []
train_Y = []
for features,label in train_data:
    train_X.append(features)
    train_Y.append(label)
test_X = []
test_Y = []
for features,label in test_data:
    test_X.append(features)
    test_Y.append(label)
train_X = np.array(train_X)/255.0
train_X = train_X.reshape(-1,64,64,3)
train_Y = np.array(train_Y)
test_X = np.array(test_X)/255.0
test_X = test_X.reshape(-1,64,64,3)
test_Y = np.array(test_Y)
w=10
h=10
fig=plt.figure(figsize=(12,12))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    img = train_X[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(np.squeeze(img))
plt.show()
def relu_bn(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn
def residual_block(x, downsample: bool, filters: int, kernel_size: int = 3):
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out
inputs = Input(shape=(64, 64, 3))
num_filters = 32
    
t = BatchNormalization()(inputs)
t = Conv2D(kernel_size=3,
           strides=1,
           filters=32,
           padding="same")(t)
t = relu_bn(t)
    
num_blocks_list = [1, 3, 5, 6, 1]
for i in range(len(num_blocks_list)):
    num_blocks = num_blocks_list[i]
    for j in range(num_blocks):
        t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
    num_filters *= 2
    
t = AveragePooling2D(4)(t)
t = Flatten()(t)
outputs = Dense(1, activation='sigmoid')(t)
    
model = Model(inputs, outputs)
model.summary()
model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
history=model.fit(train_X,train_Y,batch_size = 64,epochs=200,validation_split = 0.1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
score = model.evaluate(test_X, test_Y, verbose=0)
print("Loss: " + str(score[0]))
print("Accuracy: " + str(score[1]*100) + "%")