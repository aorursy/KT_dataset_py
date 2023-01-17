import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from keras.applications import DenseNet121

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import keras

import csv

import gc

import cv2

from tqdm import tqdm_notebook



train_csv = "../input/aptos2019-blindness-detection/train.csv"

test_csv = "../input/aptos2019-blindness-detection/test.csv"

train_dir = "../input/aptos2019-blindness-detection/train_images/"

test_dir = "../input/aptos2019-blindness-detection/test_images/"
df = pd.read_csv(train_csv) 

size = 256,256 # input image size
# cropping function (uses edge detection to crop images)

def get_cropped_image(image):

    img = cv2.blur(image,(2,2))

    slice1Copy = np.uint8(img)

    canny = cv2.Canny(slice1Copy, 0, 50)

    pts = np.argwhere(canny>0)

    y1,x1 = pts.min(axis=0)

    y2,x2 = pts.max(axis=0)

    cropped_img = img[y1:y2, x1:x2]

    cropped_img = cv2.resize(cropped_img, size)

    return cropped_img
sample_to_show = ['07419eddd6be.png','0124dffecf29.png']



def get_cropped_image_demo(image):

    img = cv2.blur(image,(2,2))

    slice1Copy = np.uint8(img)

    canny = cv2.Canny(slice1Copy, 0, 50)

    pts = np.argwhere(canny>0)

    y1,x1 = pts.min(axis=0)

    y2,x2 = pts.max(axis=0)

    cropped_img = img[y1:y2, x1:x2]

    return np.array(cropped_img)



names = []

samples = []

cropped_images = []

for i in sample_to_show:

    path = train_dir + str(i)

    img_ = cv2.imread(path)

    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

    samples.append(img_)

    cropped_ = get_cropped_image_demo(img_)

    cropped_images.append(cropped_)

    

fig = plt.figure(figsize = (5,5))

ax1 = fig.add_subplot(2,2,1)

ax1.title.set_text('original image'), ax1.axis("off"), plt.imshow(samples[0])

ax2 = fig.add_subplot(2,2,2)

ax2.title.set_text('cropped image'), ax2.axis("off"), plt.imshow(cropped_images[0])

ax3 = fig.add_subplot(2,2,3)

ax3.title.set_text('original image'), ax3.axis("off"), plt.imshow(samples[1])

ax4 = fig.add_subplot(2,2,4)

ax4.title.set_text('cropped image'), ax4.axis("off"), plt.imshow(cropped_images[1]);
def load_image(path):

    img = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), size)

    img = get_cropped_image(img)

    return img
training_paths = [train_dir + str(x) + str(".png") for x in df["id_code"]]

images = np.empty((len(df), 256,256,3), dtype = np.uint8)

for i, path in tqdm_notebook(enumerate(training_paths)):

    images[i,:,:,:] = load_image(path)
labels = df["diagnosis"].values.tolist()

labels = keras.utils.to_categorical(labels)
images, x_val, labels, y_val = train_test_split(images, labels, test_size = 0.15)
train_aug = ImageDataGenerator(horizontal_flip = True,

                               zoom_range = 0.25,

                               rotation_range = 360,

                               vertical_flip = True)



train_generator = train_aug.flow(images, labels, batch_size = 8)
input_layer = Input(shape = (256,256,3))

base_model = DenseNet121(include_top = False, input_tensor = input_layer, weights = "../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")

x = GlobalAveragePooling2D()(base_model.output)

x = Dropout(0.5)(x)

out = Dense(5, activation = 'softmax')(x)



model = Model(inputs = input_layer, outputs = out)
optimizer = keras.optimizers.Adam(lr=3e-4)



es = EarlyStopping(monitor='val_loss', mode='min', patience = 5, restore_best_weights = True)

rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience = 2, factor = 0.5, min_lr=1e-6)

    

callback_list = [es, rlrop]



model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"]) 
model.fit_generator(generator = train_generator, steps_per_epoch = len(train_generator), epochs = 20, validation_data = (x_val, y_val), callbacks = callback_list)
del train_generator, images

gc.collect()
test_df = pd.read_csv(test_csv)

test_paths = [test_dir + str(x) + str(".png") for x in test_df["id_code"]]

test_images = np.empty((len(test_df), 256,256,3), dtype = np.uint8)

for i, path in tqdm_notebook(enumerate(test_paths)):

    test_images[i,:,:,:] = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), size)
predprobs = model.predict(test_images)
predictions = []

for i in predprobs:

    predictions.append(np.argmax(i)) 
id_code = test_df["id_code"].values.tolist()

subfile = pd.DataFrame({"id_code":id_code, "diagnosis":predictions})

subfile.to_csv('submission.csv',index=False)