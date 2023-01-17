from tensorflow_core.python.keras.utils import np_utils
import json
from sklearn.utils import shuffle
import cv2
import random
import numpy as np
import tensorflow as tf
import time

from math import ceil
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense


import matplotlib.pyplot as plt
%matplotlib inline


#训练参数设置
EPOCHS = 20
BATCH_SIZE = 64
lEARN_RATE = 0.01

INPUT_SIZE = 224   #输入样本的维度大小
train_num = 31718  #训练集样本数
val_num = 4507     #验证集样本数

TRAIN_DIR = "../input/mydataset/train/train/images/"
TRAIN_ANNOTATION_FILE = "../input/mydataset/train/train/AgriculturalDisease_train_annotations.json"

VAL_DIR = "../input/dataset/val1/val/images/"
VAL_ANNOTATION_FILE = "../input/dataset/val1/val/AgriculturalDisease_validation_annotations.json"
#build vgg16 模型
model = Sequential()
#1
model.add(Conv2D(64, (3,3), input_shape=(224, 224, 3), padding='same', activation='relu', name='conv1_1'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu', name='conv1_2'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3,3), padding='same', activation='relu', name='conv2_1'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', activation='relu', name='conv2_2'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3,3), padding='same', activation='relu', name='conv3_1'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same', activation='relu', name='conv3_2'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same', activation='relu', name='conv3_3'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3,3), padding='same', activation='relu', name='conv4_1'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', activation='relu', name='conv4_2'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', activation='relu', name='conv4_3'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3,3), padding='same', activation='relu', name='conv5_1'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', activation='relu', name='conv5_2'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', activation='relu', name='conv5_3'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))  #4096
model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu'))  #4096
model.add(Dropout(0.5))

model.add(Dense(61, activation='softmax'))  #1000


#编译模型
sgd = SGD(lr=lEARN_RATE, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])
#utils工具文件
batch_feature = []
# 初始化sess,或回复保存的sess
def start_or_restore_training(sess, saver, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        step = 1
        print('start training from new state')
    return sess, step

# 含中文路径读取图片方法
def cv_imread(filepath):
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img

# 等比例缩放图片,size为最边短边长
def resize_img(img, size):
    h = img.shape[0]
    w = img.shape[1]
    scale = max(size / h, size / w)
    resized_img = cv2.resize(img, (int(h * scale), int(w * scale)))
    return resized_img

# 对缩放图片进行随机切割,要求输入图片其中一边与切割大小相等
def random_crop(img, size):
    h = img.shape[0]
    w = img.shape[1]
    if h > w and h > size:
        offset = random.randint(0, h - size)
        croped_img = img[offset:offset + size, :]
    elif h < w and w > size:
        offset = random.randint(0, w - size)
        croped_img = img[:, offset:offset + size]
    elif h == w and h > size:
        offset = random.randint(0, h - size)
        croped_img = img[offset:offset + size, offset:offset + size]
    else:
        croped_img = img
    return croped_img

def load_feature(img_path):
    img = cv_imread(img_path)
    norm_img = img / 255.
    crop = resize_img(norm_img, INPUT_SIZE)
    crop = random_crop(crop, INPUT_SIZE)
    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
    return crop

def process_annotation(anno_file, dir):
    with open(anno_file) as file:
        annotations = json.load(file)
        img_paths = []
        labels = []
        for anno in annotations:
            img_paths.append(dir + anno["image_id"])
            labels.append(anno["disease_class"])
    return img_paths, labels

def make_one_hot(data1):
    return (np.arange(61) == data1[:, None]).astype(np.integer)

def data_generator(img_paths, labels, batch_size, is_shuffle=True):
    if is_shuffle:
        img_paths, labels = shuffle(img_paths, labels)
    num_sample = len(img_paths)
    print(num_sample)
    while True:
        if is_shuffle:
            img_paths, labels = shuffle(img_paths, labels)

        for offset in range(0, num_sample, batch_size):
            batch_paths = img_paths[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            batch_labels = np.array(batch_labels)

            batch_features = [load_feature(path) for path in batch_paths]
            batch_labels = np_utils.to_categorical(batch_labels, num_classes=61)

            batch_feature = np.array(batch_features)
            yield batch_feature, batch_labels

def validation(sess, acc, x, y, rate, anno_file, dir, batch_size):
    img_paths, labels = process_annotation(anno_file, dir)
    data_gen = data_generator(img_paths, labels, batch_size, is_shuffle=False)
    num_sample = len(img_paths)
    num_it = ceil(num_sample / batch_size)

    total_accuracy = 0

    for i in range(num_it):
        features, labels = next(data_gen)
        accuracy = sess.run(acc, feed_dict={x: features, y: labels, rate: 1.0})
        total_accuracy += accuracy

    return total_accuracy / num_it

trian_img_paths, train_labels = process_annotation(TRAIN_ANNOTATION_FILE, TRAIN_DIR)
trian_img_paths1, train_labels1 = process_annotation(VAL_ANNOTATION_FILE, VAL_DIR)

# 训练模型
tic = time.time()
history = model.fit_generator(data_generator(trian_img_paths, train_labels, BATCH_SIZE),
                              steps_per_epoch=train_num // BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=data_generator(trian_img_paths1, train_labels1, BATCH_SIZE),
                              validation_steps=val_num // BATCH_SIZE,
                              shuffle=True,
                              verbose=1)
toc = time.time()
print("")
print('used time:',toc - tic,'\n')
#模型保存
model.save_weights("vgg16_weight_v12.h5")
789101112131415161718191120
#保存并显示训练性能图
title = "Crop Disease Identification Vgg16 Training Performance"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(title,fontsize=12)
fig.subplots_adjust(top=0.85,wspace=0.3)
epoch_list = list(range(1, EPOCHS +1))
ax1.plot(epoch_list,history.history['loss'], color='b', label="Training loss")
ax1.plot(epoch_list,history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(0, EPOCHS+1, 1))
ax1.set_ylabel('Loss Value')
ax1.set_xlabel('Epoch#')
ax1.set_title('Loss')
ax1.legend(loc="best")

ax2.plot(epoch_list,history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(epoch_list,history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(0, EPOCHS+1, 1))
ax2.set_ylabel('Accuracy Value')
ax2.set_xlabel('Epoch#')
ax2.set_title('Accuracy')
ax2.legend(loc="best")

plt.savefig("Vgg16_Training_Performance_v12.png")
plt.tight_layout()
plt.show()