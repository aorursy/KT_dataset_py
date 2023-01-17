import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import glob

import os
os.listdir('../input/cat-and-dog/training_set/training_set/')         # Kaggle datasets路径
train_image_path = glob.glob('../input/cat-and-dog/training_set/training_set/*/*.*')
len(train_image_path)
train_image_path[-5:]
p = '../input/cat-and-dog/training_set/training_set/cats/cat.2278.jpg'
p.split('/')
p.split('/')[5]
p.split('/')[5] == 'cats'
int(p.split('/')[5] == 'cats')
train_image_label = [int(p.split('/')[5] == 'cats') for p in train_image_path]  #cat=1 dog=0
train_image_label[0:5]
train_image_path[0:5]
def load_preprocess_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [360, 360])

    image = tf.image.random_crop(image, [256, 256, 3])                      # 随机裁减

    image = tf.image.random_flip_left_right(image)                          # 随机左右翻转

    image = tf.image.random_flip_up_down(image)                             # 随机上下翻转

#     image = tf.image.random_brightness(image, 0.5)                          # 随机亮度

#     image = tf.image.random_contrast(image, 0, 1)                           # 随机对比度

#     image = tf.image.random_hue(image, max_delta=0.3)                       # 随机颜色

#     image = tf.image.random_saturation(image,lower=0.2, upper=1.8)          # 随机饱和度

    

    image = tf.cast(image, tf.float32)

    image = image/255.0

    label = tf.reshape(label, [1])       #[1,2,3] -->[[1],[2],[3]]

    return image, label
# tf.image.convert_image_dtype
train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_image_ds = train_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
train_image_ds
BATCH_SIZE = 32

train_count = len(train_image_path)
train_count
train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)
train_image_ds = train_image_ds.prefetch(AUTOTUNE)
#创建测试集
def load_preprocess_test_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [256, 256])

    image = tf.cast(image, tf.float32)

    image = image/255.0

    label = tf.reshape(label, [1])       #[1,2,3] -->[[1],[2],[3]]

    return image, label
test_image_path = glob.glob('../input/cat-and-dog/test_set/test_set/*/*.*')

test_image_label = [int(p1.split('/')[5] == 'cats') for p1 in test_image_path]

test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))

test_image_ds = test_image_ds.map(load_preprocess_test_image, num_parallel_calls=AUTOTUNE)

test_image_ds = test_image_ds.batch(BATCH_SIZE)

test_image_ds = test_image_ds.prefetch(AUTOTUNE)
test_image_label[-5:]
test_image_path[-5:]
imgs, labels = next(iter(train_image_ds))
imgs.shape
labels.shape
plt.imshow(imgs[12])
labels[0]
# 类VGG-16

model= keras.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(None, None, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),



    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, (1, 1), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(512, (1, 1), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    

    

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(512, (1, 1), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    

    tf.keras.layers.GlobalAveragePooling2D(),

    

    tf.keras.layers.Dense(4096, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(4096, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1000, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1)

])
model.summary()
pred = model(imgs)
pred.shape
np.array([p[0].numpy() for p in tf.cast(pred > 0, tf.int32)])
np.array([l[0].numpy() for l in labels])
ls = tf.keras.losses.BinaryCrossentropy()
ls([0.,0.,1.,1.], [1.,1.,1.,1.])
ls([[0.],[0.],[1.],[1.]], [[1.],[1.],[1.],[1.]])
tf.keras.losses.binary_crossentropy([0.,0.,1.,1.], [1.,1.,1.,1.])
optimizer = tf.keras.optimizers.Adam()
epoch_loss_avg = tf.keras.metrics.Mean('train_loss')     # 记录平均损失

train_accuracy = tf.keras.metrics.Accuracy()            # 记录正确率
epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')     # 记录平均损失

test_accuracy = tf.keras.metrics.Accuracy()            # 记录正确率
train_accuracy([1,0,1], [1,1,1])
def train_step(model, imgs, labels):

    with tf.GradientTape() as t:

        pred = model(imgs)    # 计算预测值

        loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)    # 计算损失

    grads = t.gradient(loss_step, model.trainable_variables)   # 计算损失和可训练参数之间的梯度

    optimizer.apply_gradients(zip(grads,model.trainable_variables))   # 优化 

    epoch_loss_avg(loss_step)

    train_accuracy(labels, tf.cast(pred>0, tf.int32))
def test_step(model, imgs, labels):

    pred = model(imgs, training=False)    # 计算预测值

    loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)    # 计算损失

    epoch_loss_avg_test(loss_step)

    test_accuracy(labels, tf.cast(pred>0, tf.int32))
train_loss_results = []

train_acc_results = []



test_loss_results = []

test_acc_results = []
num_epochs = 30
for epoch in range(num_epochs):    # 定义单批次训练函数

    for imgs_, labels_ in train_image_ds:

        train_step(model, imgs_, labels_)

        print('.', end='')

    print()   #换行

    

    train_loss_results.append(epoch_loss_avg.result())  # 记录损失

    train_acc_results.append(train_accuracy.result())    # 记录正确率

    

    for imgs_, labels_ in test_image_ds:

        test_step(model, imgs_, labels_)

    test_loss_results.append(epoch_loss_avg_test.result())  # 记录损失

    test_acc_results.append(test_accuracy.result())    # 记录正确率

    

    

    print('Epoch:{}:loss:{:.3f},accuracy:{:.3f}, test_loss:{:.3f},test_accuracy:{:.3f}'.format(

        epoch + 1,

        epoch_loss_avg.result(),

        train_accuracy.result(),

        epoch_loss_avg_test.result(),

        test_accuracy.result()

    ))

    

    epoch_loss_avg.reset_states()         # 初始化

    train_accuracy.reset_states()         # 初始化

    epoch_loss_avg_test.reset_states()   # 初始化

    test_accuracy.reset_states()         # 初始化