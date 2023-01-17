from __future__ import absolute_import, division, print_function
%matplotlib inline
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
import os
import sys
import random
import numpy as np
import cv2
from tqdm import tqdm

seed = 42
random.seed = seed
np.random.seed = seed

# Set some parameters
DATA_PATH = '../input'
TRAIN_PATH = DATA_PATH + '/stage1_train/'
TEST_PATH  = DATA_PATH + '/stage1_test/'

IMAGE_W = 128
IMAGE_H = 128
IMAGE_C = 1
# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids  = next(os.walk(TEST_PATH))[1]
len(train_ids), len(test_ids)
# Get and resize train images and masks
def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(IMAGE_W,IMAGE_H))
    img = img.astype(np.float32)/255.0
    img = np.expand_dims(img, axis=-1)
    return img

def load_data(train_path, test_path, shuffle=False):
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()

    trainX = np.zeros((len(train_ids), IMAGE_H, IMAGE_W, IMAGE_C), dtype=np.float32)
    trainY = np.zeros((len(train_ids), IMAGE_H, IMAGE_W, 1), dtype=np.bool)
    for i, name in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = train_path + name
        trainX[i] = load_image(path + '/images/' + name + '.png')

        # 将多个分开的掩码合在一起，边界地方的像素点值为 True
        mask = np.zeros((IMAGE_H, IMAGE_W, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            img = cv2.imread(path + '/masks/' + mask_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(IMAGE_W,IMAGE_H))
            img = np.expand_dims(img, axis=-1)
            mask = np.maximum(mask, img)
        trainY[i] = mask

    # Get and resize test images
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    testX = np.zeros((len(test_ids), IMAGE_H, IMAGE_W, IMAGE_C), dtype=np.float32)
    sizes_test = []
    for i, name in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = test_path + name
        testX[i] = load_image(path + '/images/' + name + '.png')
    
    if shuffle:
        trainX, trainY = shuffle_data(trainX, trainY)

    return trainX, trainY, testX

trainX, trainY, testX = load_data(TRAIN_PATH, TEST_PATH)
trainX.shape, trainY.shape, testX.shape
def image_augmentation(trainX, trainY):
    n_imgaug = 1+2 # 1 origin + 2 augmentation
    shape = trainX.shape
    shape = (trainX.shape[0]*n_imgaug,) + trainX.shape[1:] 
    new_trainX = np.zeros(shape, dtype=np.float32)
    new_trainY = np.zeros(shape, dtype=np.float32)
    for i in tqdm(range(len(trainX))):
        img = trainX[i]
        new_trainX[i*n_imgaug+0] = img
        new_trainX[i*n_imgaug+1] = np.fliplr(img)
        new_trainX[i*n_imgaug+2] = np.flipud(img)
        
        img = trainY[i]
        new_trainY[i*n_imgaug+0] = img
        new_trainY[i*n_imgaug+1] = np.fliplr(img)
        new_trainY[i*n_imgaug+2] = np.flipud(img)

    return new_trainX, new_trainY

trainX, trainY = image_augmentation(trainX, trainY)
trainX.shape, trainY.shape
# Check if training data looks all right
ix = random.randint(0, len(trainX)-1)
image = trainX[ix].reshape(IMAGE_H, IMAGE_W)

plt.figure(figsize=(10, 10))

plt.subplot(121)
plt.grid(False)
plt.imshow(image, plt.cm.gray)
plt.title("Image")

plt.subplot(122)
plt.grid(False)
plt.imshow(np.squeeze(trainY[ix]), plt.cm.gray)
plt.title("Mask");
del image
import time
import tflearn
import tensorflow as tf
from tflearn import input_data, dropout, fully_connected
from tflearn import conv_2d, max_pool_2d, conv_2d_transpose, upsample_2d
from tflearn import merge
from tflearn import regression
from tflearn import ImagePreprocessing
from tflearn import ImageAugmentation
from tflearn import Momentum

tf.reset_default_graph()

d0 = input_data(shape=[None, IMAGE_H, IMAGE_W, IMAGE_C], name="input")

c1 = conv_2d(d0,  8, 3, weights_init='variance_scaling', activation='relu', name="conv1_1")
c1 = conv_2d(c1,  8, 3, weights_init='variance_scaling', activation='relu', name="conv1_2")
p1 = max_pool_2d(c1, 2)

c2 = conv_2d(p1, 16, 3, weights_init='variance_scaling', activation='relu', name="conv2_1")
c2 = conv_2d(c2, 16, 3, weights_init='variance_scaling', activation='relu', name="conv2_2")
p2 = max_pool_2d(c2, 2)

c3 = conv_2d(p2, 32, 3, weights_init='variance_scaling', activation='relu', name="conv3_1")
c3 = conv_2d(c3, 32, 3, weights_init='variance_scaling', activation='relu', name="conv3_2")
p3 = max_pool_2d(c3, 2)

c4 = conv_2d(p3, 64, 3, weights_init='variance_scaling', activation='relu', name="conv4_1")
c4 = conv_2d(c4, 64, 3, weights_init='variance_scaling', activation='relu', name="conv4_2")
p4 = max_pool_2d(c4, 2)

c5 = conv_2d(p4, 128, 3, weights_init='variance_scaling', activation='relu', name="conv5_1")
c5 = conv_2d(c5, 128, 3, weights_init='variance_scaling', activation='relu', name="conv5_2")

u6 = conv_2d_transpose(c5, 64, 2, [ 16, 16], strides=2)
u6 = merge([u6, c4], mode='concat', axis=3, name='upsamle-5-merge-4')
c6 = conv_2d(u6, 64, 3, weights_init='variance_scaling', activation='relu', name="conv6_1")
c6 = conv_2d(c6, 64, 3, weights_init='variance_scaling', activation='relu', name="conv6_2")

u7 = conv_2d_transpose(c6, 32, 2, [ 32, 32], strides=2)
u7 = merge([u7, c3], mode='concat', axis=3, name='upsamle-6-merge-3')
c7 = conv_2d(u7, 32, 3, weights_init='variance_scaling', activation='relu', name="conv7_1")
c7 = conv_2d(c7, 32, 3, weights_init='variance_scaling', activation='relu', name="conv7_2")

u8 = conv_2d_transpose(c7, 16, 2, [ 64, 64], strides=2)
u8 = merge([u8, c2], mode='concat', axis=3, name='upsamle-7-merge-2')        
c8 = conv_2d(u8, 16, 3, weights_init='variance_scaling', activation='relu', name="conv8_1")
c8 = conv_2d(c8, 16, 3, weights_init='variance_scaling', activation='relu', name="conv8_2")

u9 = conv_2d_transpose(c8,  8, 2, [128,128], strides=2)
u9 = merge([u9, c1], mode='concat', axis=3, name='upsamle-8-merge-1')
c9 = conv_2d(u9,  8, 3, weights_init='variance_scaling', activation='relu', name="conv9_1")
c9 = conv_2d(c9,  8, 3, weights_init='variance_scaling', activation='relu', name="conv9_2")

fc = conv_2d(c9,  1, 1, weights_init='variance_scaling', activation='linear', name="target")

# Define IoU metric
def mean_iou_accuracy_op(y_pred, y_true, x):
    with tf.name_scope('Accuracy'):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_tmp = tf.to_int32(y_pred > 0.5)
            score, update_op = tf.metrics.mean_iou(y_true, y_pred_tmp, 2)
            with tf.Session() as sess:
                sess.run(tf.local_variables_initializer())
            with tf.control_dependencies([update_op]):
                score = tf.identity(score)
            prec.append(score)
        acc = tf.reduce_mean(tf.stack(prec), axis=0, name='mean_iou')
    return acc

net = regression(fc,
                 optimizer='Adam',
                 loss='binary_crossentropy',
                 metric=mean_iou_accuracy_op,
                 learning_rate=0.001
                )

model = tflearn.DNN(net, tensorboard_verbose=3)

start_time = time.time()
model.fit(trainX, 
          trainY, 
          validation_set=0.1,
          n_epoch=20,
          batch_size=16,
          shuffle=True,
          show_metric=True,
          run_id='bowl_unet')

duration = time.time() - start_time
print('Training Duration %.3f sec' % (duration))
ix = random.randint(0, len(testX)-1)
image = testX[ix:ix+1]
y_pred = model.predict(image)
y_pred = (y_pred > 0.5).astype(np.uint8).reshape(IMAGE_H, IMAGE_W)

plt.figure(figsize=(10, 10))

plt.subplot(121)
plt.grid(False)
plt.imshow(image.reshape(IMAGE_H, IMAGE_W), plt.cm.gray)
plt.title("Image")

plt.subplot(122)
plt.grid(False)
plt.imshow(np.squeeze(y_pred), plt.cm.gray)
plt.title("Predicted Mask");
del image,y_pred