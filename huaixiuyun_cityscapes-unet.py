# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import glob

import time

import pickle

import cv2

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')

tf.config.experimental.set_memory_growth(gpu[0], True)
! ls /kaggle/input/cityscapes/Cityspaces/images/train/aachen | head
 !ls /kaggle/input/cityscapes/Cityspaces/gtFine/train/aachen | head

    
train_image_path = "/kaggle/input/cityscapes/Cityspaces/images/train/*/*.png"

train_label_path = "/kaggle/input/cityscapes/Cityspaces/gtFine/train/*/*_gtFine_labelIds.png"

val_image_path = "/kaggle/input/cityscapes/Cityspaces/images/val/*/*.png"

val_label_path = "/kaggle/input/cityscapes/Cityspaces/gtFine/val/*/*_gtFine_labelIds.png"
train_images = sorted(glob.glob(train_image_path))

train_labels = sorted(glob.glob(train_label_path))

val_images = sorted(glob.glob(val_image_path))

val_labels = sorted(glob.glob(val_label_path))

# 这里不排序， 产生的顺序不一样，导致 image 和 label 不匹配。 但在window下不用排序
train_images[1000], train_labels[1000], val_images[400], val_labels[400]
len(train_images ), len(val_images), len(train_labels), len(val_labels)
dataset_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

dataset_val = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
def read_png(path, channels=3): 

    # path:  image or label path, if image path, channels=3.  if label path, channels=1

    # because in this training, we read label is  gtFine_labelIds.png(channels=1) ,  not _gtFine_color.png

    img = tf.io.read_file(path)

    img = tf.image.decode_png(img, channels=channels)

    return img



def crop_img(img, label):

    concat_img = tf.concat([img, label], axis=-1)

    concat_img = tf.image.resize(concat_img, (280, 280), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    crop_img = tf.image.random_crop(concat_img, [256, 256, 4])

    # tf.image.random_crop， crop a tensor to a given size randomly, here is [256, 256, 4]

    # images and labels are cropped at the same time to maintain consistency,  so need to concat([img, label]) 

    return crop_img[:, :, 0:3], crop_img[:, :, 3:]



def normal(img, label):

    img = tf.cast(img, tf.float32)/127.5 -1

    label = tf.cast(label, tf.int32)

    return img, label
def load_image_train(img_path, label_path):

    img = read_png(img_path)

    label = read_png(label_path, channels=1)

    img, label = crop_img(img, label)

    if tf.random.uniform(()) > 0.5:

        img = tf.image.flip_left_right(img)

        label = tf.image.flip_left_right(label)           

    img, label = normal(img, label)

    return img, label



def load_image_val(img_path, label_path):

    

    img = read_png(img_path)

    label = read_png(label_path, channels=1)

    

    img = tf.image.resize(img, (256, 256))

    label = tf.image.resize(label, (256, 256))

    

    img, label = normal(img, label)

    return img, label
index = np.random.permutation(len(train_images))

train_images = np.array(train_images)[index]

train_labels = np.array(train_labels)[index]
BATCH_SIZE = 32

BUFFER_SIZE = 300

EPOCHS = 60

train_count = len(train_images)

val_count = len(val_images)

train_step_per_epoch = train_count // BATCH_SIZE

val_step_per_epoch = val_count // BATCH_SIZE

auto = tf.data.experimental.AUTOTUNE
dataset_train = dataset_train.map(load_image_train, num_parallel_calls=auto)

dataset_val =dataset_val.map(load_image_val, num_parallel_calls=auto)



dataset_train = dataset_train.cache().repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(auto)

dataset_val = dataset_val.cache().batch(BATCH_SIZE)
for image, label in dataset_train.take(1):

    plt.figure(figsize=(10, 10))

    plt.subplot(121)

    plt.title('image')

    plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))

    plt.subplot(122)

    plt.title('label')

    plt.imshow(tf.keras.preprocessing.image.array_to_img(label[0])) 
for image, label in dataset_val.take(1):



    plt.figure(figsize=(10, 10))

    plt.subplot(121)

    plt.title('image')

    plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))

    plt.subplot(122)

    plt.title('label')

    plt.imshow(tf.keras.preprocessing.image.array_to_img(label[0])) 
def create_model():

    inputs = tf.keras.layers.Input(shape=(256, 256, 3))

    

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)

    x = tf.keras.layers.BatchNormalization()(x)    

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = tf.keras.layers.BatchNormalization()(x)

 

    #   x shape  (None, 256, 256, 64)

    x1 = tf.keras.layers.MaxPooling2D(padding='same')(x)

    # (None, 128, 128, 64)

    

    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)

    x1 = tf.keras.layers.BatchNormalization()(x1)    

    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)

    x1= tf.keras.layers.BatchNormalization()(x1)

    #  shape  (None, 128, 128, 128)

        

    x2 = tf.keras.layers.MaxPooling2D(padding='same')(x1)

    # shape N(one, 64, 64, 128)

    

    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)

    x2 = tf.keras.layers.BatchNormalization()(x2)    

    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)

    x2= tf.keras.layers.BatchNormalization()(x2)

    #  shape  (None, 64, 64, 256)  

    

    x3 = tf.keras.layers.MaxPooling2D(padding='same')(x2)

    #     shape  (None, 32, 32, 256) 

    

    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)

    x3 = tf.keras.layers.BatchNormalization()(x3)    

    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)

    x3= tf.keras.layers.BatchNormalization()(x3)

    #  shape  (None, 32, 32, 512)  

    

    x4 = tf.keras.layers.MaxPooling2D(padding='same')(x3)

    #     shape  (None, 16, 16, 512) 

    

    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)

    x4 = tf.keras.layers.BatchNormalization()(x4)    

    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)

    x4= tf.keras.layers.BatchNormalization()(x4)

    #  shape  (None, 16, 16, 1024)    

    

    # 上采样

    x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding='same',

                                        activation='relu')(x4)

    x5 = tf.keras.layers.BatchNormalization()(x5)

    #  shape  (None, 32, 32, 512)  , 和 x3  shape一样

#     print("x3, x5 shape:", x3.shape, x5.shape)

    x6 = tf.concat([x3, x5], axis=-1)

    #  shape  (None, 32, 32, 1024)

    

    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)

    x6 = tf.keras.layers.BatchNormalization()(x6)    

    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)

    x6= tf.keras.layers.BatchNormalization()(x6)

    #    (None, 32, 32, 512)

    

    x7= tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same',

                                        activation='relu')(x6)

    x7 = tf.keras.layers.BatchNormalization()(x7)

    #  shape  (None, 64, 64, 256)  和 x2 shape 一样， concatenate起来  

#     print("x2, x7 shape:", x2.shape, x7.shape)

    x8 = tf.concat([x2, x7], axis=-1)

    #   (None, 64, 64, 512)

    

    

    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)

    x8 = tf.keras.layers.BatchNormalization()(x8)    

    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)

    x8= tf.keras.layers.BatchNormalization()(x8)

    #    (None, 64, 64, 256)

    

    x9= tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same',

                                        activation='relu')(x8)

    x9 = tf.keras.layers.BatchNormalization()(x9)

    #   (None, 128, 128, 128)

#     print("x1, x9 shape:", x1.shape, x9.shape)

    x10 = tf.concat([x1, x9], axis=-1)

    #   (None, 128, 128, 256)

    

    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)

    x10 = tf.keras.layers.BatchNormalization()(x10)    

    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)

    x10 = tf.keras.layers.BatchNormalization()(x10)

    #    (None, 128, 128, 128)  

    

    

    x11= tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same',

                                        activation='relu')(x10)

    x11 = tf.keras.layers.BatchNormalization()(x11)

    #   (None, 256, 256, 64)   和 x shape一样

#     print("x, x11 shape:", x.shape, x11.shape)

    x11 = tf.concat([x, x11], axis=-1)

    #   (None, 256, 256, 128) 

    

    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x11)

    x12 = tf.keras.layers.BatchNormalization()(x12)    

    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x12)

    x12 = tf.keras.layers.BatchNormalization()(x12)

    #    (None, 256, 256, 64) 

    

    output = tf.keras.layers.Conv2D(34, 1, padding='same', activation='softmax')(x12)

    #   34 为这个数据集Label的类别数，  shape  (None, 256, 256, 34) , 

    #最后就是算 各个channel的最大，就是某一点所属的类别

    

    return tf.keras.Model(inputs=inputs, outputs=output)
model = create_model()
model.summary()
tf.keras.utils.plot_model(model, to_file='model.png')
#tf.keras.metrics.MeanIoU(34)   #tf.keras.metrics.MeanIoU(num_classes), 参数是类别数

#但由于这个函数 使用的是独热编码方式的类别表示方法， 而本例中的类别数是稀疏表示的，并非独热编码，因此需要对函数做 部分更改

class MeanIoU(tf.keras.metrics.MeanIoU):

#     def __call__(self, y_true, y_pred, sample_weight=None):

#         y_pred = tf.argmax(y_pred, axis=-1)

#         return super().__call__(y_true, y_pred, sample_weight=sample_weight)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.argmax(y_pred, axis=-1)

        return super().update_state(y_true, y_pred, sample_weight)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc',

                                                                                MeanIoU(num_classes=34)])
dataset_train, dataset_val
start = time.time()

history = model.fit(dataset_train, epochs=EPOCHS, 

                   steps_per_epoch=train_step_per_epoch,

                   validation_steps=val_step_per_epoch,

                   validation_data=dataset_val)

end =time.time()

print(str(int(end - start)))
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(EPOCHS)

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss Value')

plt.legend()

plt.show()
num = 3

for image, mask in dataset_val.take(1):

    pred_mask = model.predict(image)

    pred_mask = tf.argmax(pred_mask, axis=-1)

    pred_mask = pred_mask[..., tf.newaxis]

    

    plt.figure(figsize=(10, 10))

    for i in range(num):

        plt.subplot(num, 3, i*num+1)

        plt.title('real image')

        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]))

        plt.subplot(num, 3, i*num+2)

        plt.title('real label')

        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))

        plt.subplot(num, 3, i*num+3)

        plt.title('pred label')

        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))
for image, mask in dataset_train.take(1):

    pred_mask = model.predict(image)

    pred_mask = tf.argmax(pred_mask, axis=-1)

    pred_mask = pred_mask[..., tf.newaxis]

    

    plt.figure(figsize=(10, 10))

    for i in range(num):

        plt.subplot(num, 3, i*num+1)

        plt.title('real image')

        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]))

        plt.subplot(num, 3, i*num+2)

        plt.title('real label')

        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))

        plt.subplot(num, 3, i*num+3)

        plt.title('pred label')

        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))
odel.save('unet_v7.h5')