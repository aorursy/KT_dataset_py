from tensorflow_core.python.keras.utils import np_utils
import json
from sklearn.utils import shuffle
import cv2
import random
import numpy as np
import tensorflow as tf
import time

from math import ceil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense


import matplotlib.pyplot as plt
%matplotlib inline


#训练参数设置
EPOCHS = 15
BATCH_SIZE = 128
lEARN_RATE = 0.001

img_height = 224
img_width = 224   #输入样本的维度大小

train_num = 31718  #训练集样本数
val_num = 4507     #验证集样本数

# TRAIN_DIR = "../input/mydataset/train/train/images"
TRAIN_DIR = "../input/mydataclass/trainimg/trainimg/"

TRAIN_ANNOTATION_FILE = "../input/mydataset/train/train/AgriculturalDisease_train_annotations.json"

# VAL_DIR = "../input/dataset/val1/val/images"
VAL_DIR = "../input/mydataclass/valimg/valimg/"

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

model.load_weights('../input/vgg16-weight-v13/vgg16_weight_v133.h5')
#图片进行图像预处理，增加图片归一化、适度旋转、随机缩放、上下翻转
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical') #多分类

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical') #多分类
#编译模型
sgd = SGD(lr=lEARN_RATE, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])
# 训练模型
tic = time.time()
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_num // BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=validation_generator,
                              validation_steps=val_num // BATCH_SIZE,
                              shuffle=True,
                              verbose=1)
toc = time.time()
print("")
print('used time:',toc - tic,'\n')
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

#模型保存
model.save_weights("vgg16_weight_v14.h5")
# val_accuracy = [0.3167, 0.6199, 0.7088, 0.5436, 0.7634, 0.7241, 0.7220, 0.7560, 0.7461, 0.7610, 
#  0.7835, 0.7965, 0.8283, 0.8213, 0.8229, 0.8136, 0.8152, 0.8103, 0.7763, 0.7873, 
#  0.6937, 0.8204, 0.8386, 0.8193, 0.8136, 0.8485, 0.8136, 0.8397, 0.8184, 0.8368, 
#  0.7830, 0.8242, 0.8456, 0.8332, 0.8359, 0.8085, 0.8497, 0.8445, 0.8323, 0.8382, 
#  0.8375, 0.8557, 0.8475, 0.8548, 0.8509, 0.8520, 0.8429, 0.8408, 0.8577, 0.8529, 
#  0.8436, 0.8468, 0.8564, 0.8153, 0.8545]
# loss =  [2.6408, 1.5387, 1.1869, 0.9760, 0.8495, 0.7631, 0.6846, 0.6375, 0.5911, 0.5607, 
#  0.5305, 0.5080, 0.4834, 0.4681, 0.4425, 0.4334, 0.4137, 0.4012, 0.4008, 0.3790, 
#  0.3738, 0.3621, 0.3581, 0.3422, 0.3430, 0.3312, 0.3268, 0.3263, 0.3026, 0.3026, 
#  0.3023, 0.2929, 0.2879, 0.2839, 0.2760, 0.2710, 0.2681, 0.2622, 0.2538, 0.2472, 
#  0.210066792875649, 0.1990358798224249, 0.19377904673492882, 0.18695065474615855,
#  0.18858622751986162, 0.1774076981469399, 0.1823530440392393, 0.1760597648321356,
#  0.16887169702779872, 0.170848973874202, 0.16266282304871418, 0.1613719401224839, 
#  0.16172850692562757, 0.16247937276077332, 0.1534303302797346]
# val_loss =  [2.8546, 1.2710, 0.9208, 1.7031, 0.6650, 0.7849, 0.8730, 0.6729, 0.7146, 0.6669, 
#  0.5876, 0.5878, 0.4554, 0.4691, 0.4637, 0.5091, 0.4929, 0.5242, 0.6390, 0.6225, 
#  0.9842, 0.4752, 0.4388, 0.5081, 0.5063, 0.4146, 0.5284, 0.4301, 0.5359, 0.4224, 
#  0.6305, 0.4972, 0.4409, 0.4566, 0.4393, 0.5238, 0.4234, 0.4376, 0.4679, 0.4561, 
#  0.46065278138433186, 0.4326350586754935, 0.44202321256910054, 0.42920357116631097, 
#  0.44303998351097107, 0.4481317328555243, 0.45017342056546894, 0.479198910508837,
#  0.40899807172162195, 0.4469689360686711, 0.48563805988856723, 0.4523181987660272, 
#  0.4516772721494947, 0.5677765556744166, 0.4585575827530452]
# accuracy =  [0.3445, 0.5596, 0.6344, 0.6836, 0.7151, 0.7392, 0.7615, 0.7724, 0.7858, 0.7939, 
#  0.8028, 0.8075, 0.8175, 0.8214, 0.8278, 0.8318, 0.8366, 0.8438, 0.8424, 0.8514, 
#  0.8511, 0.8589, 0.8595, 0.8613, 0.8620, 0.8679, 0.8689, 0.8685, 0.8779, 0.8765,
#  0.8772, 0.8809, 0.8829, 0.8859, 0.8862, 0.8899, 0.8893, 0.8930, 0.8981, 0.8983, 
#  0.9131, 0.9191, 0.9191, 0.9222, 0.9228, 0.9265, 0.9243, 0.9263, 0.9290, 0.9298, 
#  0.9326, 0.9338, 0.9338, 0.9339, 0.9378]

# EPOCHS = 55
#保存并显示训练性能图
title = "Crop Disease Identification Vgg16 Training Performance"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(title,fontsize=12)
fig.subplots_adjust(top=0.85,wspace=0.3)
epoch_list = list(range(1, EPOCHS +1))

# ax1.plot(epoch_list,loss, color='b', label="Training loss")
# ax1.plot(epoch_list,val_loss, color='r', label="validation loss")

ax1.plot(epoch_list,history.history['loss'], color='b', label="Training loss")
ax1.plot(epoch_list,history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(0, EPOCHS+1, 5))
ax1.set_ylabel('Loss Value')
ax1.set_xlabel('Epoch#')
ax1.set_title('Loss')
ax1.legend(loc="best")

# ax2.plot(epoch_list,accuracy, color='b', label="Training accuracy")
# ax2.plot(epoch_list,val_accuracy, color='r',label="Validation accuracy")

ax2.plot(epoch_list,history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(epoch_list,history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(0, EPOCHS+1, 5))
ax2.set_ylabel('Accuracy Value')
ax2.set_xlabel('Epoch#')
ax2.set_title('Accuracy')
ax2.legend(loc="best")

plt.savefig("Vgg16_Training_Performance.png")
plt.tight_layout()
plt.show()
