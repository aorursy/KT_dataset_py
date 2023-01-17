import tensorflow as tf

import matplotlib.pyplot as plt

import glob

import os
TRAIN_IMG_DIR = '../input/cat-and-dog/training_set/training_set/*/*.jpg'

TEST_IMG_DIR = '../input/cat-and-dog/test_set/test_set/*/*.jpg'
train_img_path = glob.glob(TRAIN_IMG_DIR)

test_img_path = glob.glob(TEST_IMG_DIR)
import random

random.shuffle(train_img_path)

random.shuffle(test_img_path)
len(train_img_path)
train_img_path[:2]
train_img_lab = [int(path.split('/')[5]=='cats') for path in train_img_path]

test_img_lab = [int(path.split('/')[5]=='cats') for path in test_img_path]
train_img_lab[-5:]
def getImgByPath_test(path,lab):

    img = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img,channels=3)

    img = tf.image.resize(img,[256,256])

    img = tf.cast(img,tf.float32)

    img = img/255

#    lab = tf.reshape(lab,[1])

    return img,lab
def getImgByPath(path,lab):

    img = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img,channels=3)

    img = tf.image.resize(img,[280,280])

    img = tf.image.random_crop(img,[256,256,3])

#    img = tf.image.random_flip_left_right(img)

#    img = tf.image.random_flip_up_down(img)

    img = tf.cast(img,tf.float32)

    img = img/255

#    lab = tf.reshape(lab,[1])

    return img,lab
train_img_ds = tf.data.Dataset.from_tensor_slices((train_img_path,train_img_lab))

test_img_ds = tf.data.Dataset.from_tensor_slices((test_img_path,test_img_lab))
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_img_ds = train_img_ds.map(getImgByPath,num_parallel_calls=AUTOTUNE)

test_img_ds = test_img_ds.map(getImgByPath_test,num_parallel_calls=AUTOTUNE)
train_img_ds
test_img_ds
BATCH = 64

TRAIN_LEN  = len(train_img_path)

train_img_ds = train_img_ds.shuffle(300).batch(BATCH)

test_img_ds = test_img_ds.batch(BATCH)
train_img_ds = train_img_ds.prefetch(AUTOTUNE)

test_img_ds = test_img_ds.prefetch(AUTOTUNE)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(256,(3,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(256,(3,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(256,(1,1),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(512,(1,1),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(512,(1,1),activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(4096,activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(4096,activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(1000,activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.summary()
loss = tf.keras.losses.BinaryCrossentropy()

optimizer = tf.keras.optimizers.Adam(0.000001)
train_loss_avg = tf.keras.metrics.Mean('loss')

train_acc = tf.keras.metrics.Accuracy()

test_loss_avg = tf.keras.metrics.Mean('test_loss')

test_acc = tf.keras.metrics.Accuracy()
def train_setp(model,imgs,labs):

    with tf.GradientTape() as t:

        prognosis = model(imgs)

        loss_setp = loss(labs,prognosis)

    grads = t.gradient(loss_setp,model.trainable_variables)

    optimizer.apply_gradients(zip(grads,model.trainable_variables))

    train_loss_avg(loss_setp)

    train_acc(labs,tf.cast(prognosis>0.5,tf.int32))
def test_setp(model,imgs,labs):

    prognosis = model.predict(imgs)

    loss_setp = loss(labs,prognosis)

    test_loss_avg(loss_setp)

    test_acc(labs, tf.cast(prognosis>0.5, tf.int32))
train_loss_res = []

train_acc_res = []

test_loss_res = []

test_acc_res = []
train_epochs = 150
for epoch in range(train_epochs):

    print('epoch:{}\n'.format(epoch+1))

    for imgs,labs in train_img_ds:

        train_setp(model,imgs,labs)

        print('.',end='')

    train_loss_res.append(train_loss_avg.result())

    train_acc_res.append(train_acc.result())

    print('\ntrain_loss:{:.4f},train_acc:{:.4f}'.format(train_loss_avg.result(),train_acc.result()))

    train_loss_avg.reset_states()

    train_acc.reset_states()

    for imgs,labs in test_img_ds:

        test_setp(model,imgs,labs)

        print('.',end='')

    test_loss_res.append(test_loss_avg.result())

    test_acc_res.append(test_acc.result())

    print('\ntest_loss:{:.4f},test_acc:{:.4f}'.format(test_loss_avg.result(),test_acc.result()))

    test_loss_avg.reset_states()

    test_acc.reset_states()