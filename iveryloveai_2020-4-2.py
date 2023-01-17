import glob 

import random

import numpy as np

import tensorflow as tf
def preprocessing_img(path,label):

    img = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img,3)

    img = tf.image.resize(img,[224,224])

    img = tf.cast(img/255,tf.float32)

    return (img,label)
train_path = glob.glob("../input/dogs-cats-images/dataset/training_set/*/*.jpg")

random.shuffle(train_path)

test_path = glob.glob("../input/dogs-cats-images/dataset/test_set/*/*.jpg")



TRAIN_COUNT = len(train_path)

TEST_COUNT = len(test_path)



train_label = [int(i.split("/")[-2] == "dogs") for i in train_path]

train_label = np.array(train_label,np.int32).reshape(TRAIN_COUNT,1)



test_label = [int(i.split("/")[-2] == "dogs") for i in test_path]

test_label = np.array(test_label,np.int32).reshape(TEST_COUNT,1)



train_data = tf.data.Dataset.from_tensor_slices((train_path,train_label))

test_data = tf.data.Dataset.from_tensor_slices((test_path,test_label))



AOTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 128

TRAIN_STEP = TRAIN_COUNT//BATCH_SIZE

TEST_STRP = TEST_COUNT//BATCH_SIZE



train_data = train_data.map(preprocessing_img,AOTOTUNE).repeat().shuffle(2000).batch(BATCH_SIZE)

test_data = test_data.map(preprocessing_img,AOTOTUNE).batch(BATCH_SIZE)
conv_base = tf.keras.applications.VGG16(weights="imagenet",include_top=False,)

conv_base.summary()
model = tf.keras.Sequential()

model.add(conv_base)

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(512,activation='relu'))

model.add(tf.keras.layers.Dense(256,activation='relu'))

model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.summary()
conv_base.trainable = False



model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005,),

    loss=tf.keras.losses.binary_crossentropy,

    metrics=['acc'])



#第一次训练

history = model.fit( 

    train_data,

    epochs=10,

    validation_split=TEST_STRP,

    validation_data=test_data,

    steps_per_epoch=TRAIN_STEP)
#第二次训练

#对预训练网络进行微调，解除最后三层的训练锁定

conv_base.trainable=True

for i in conv_base.layers[:-3]:

    i.trainable=False

#重新设置学习速率

model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005/10),

    loss=tf.keras.losses.binary_crossentropy,

    metrics=['acc'])



initial_epochs = 10 #已经训练的次数

fine_tune_epochs = 10 #还要训练的次数

all_epochs = initial_epochs + fine_tune_epochs  #训练总次数



#继续训练

history = model.fit(

    train_data,

    epochs=all_epochs,

    validation_data=test_data,

    initial_epoch=initial_epochs,

    steps_per_epoch=TRAIN_STEP,

    validation_steps=TEST_STRP)