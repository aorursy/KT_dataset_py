import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import glob

import os
print('Tensorflow version: {}'.format(tf.__version__))
os.listdir('../input/cat-and-dog/training_set/training_set')
train_image_path = glob.glob('../input/cat-and-dog/training_set/training_set/*/*.jpg')
len(train_image_path)
train_image_path[:5]
train_image_path[-5:]
p = '../input/cat-and-dog/training_set/training_set/cats/cat.3737.jpg'
int(p.split('training_set/training_set')[1].split('/')[0] == 'cats')
train_image_label = [int(p.split('training_set/training_set')[1].split('/')[0] == 'cats') for p in train_image_path]
train_image_label[-5:]
def load_preprosess_image(path, label): 

    image = tf.io.read_file(path) #图片途径进行读取

    image = tf.image.decode_jpeg(image, channels=3)  #对路径进行解码

    image = tf.image.resize(image, [360, 360])

    image = tf.image.random_crop(image, [256, 256, 3])

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_brightness(image, 0.5)

    image = tf.image.random_contrast(image, 0, 1)

    image = tf.cast(image, tf.float32)#图像进行转化格式，进行归一化



    image = image/255

    label = tf.reshape(label, [1])

    return image, label
#[1, 2, 3]  -->  [[1], [2], [3]]
#tf.image.convert_image_dtype()#非float32类型会进行归一化处理
train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
AUTOTUNE = tf.data.experimental.AUTOTUNE #自动使用并行运算
train_image_ds = train_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
train_image_ds
for img, label in train_image_ds.take(1):

    plt.imshow(img)
BATCH_SIZE = 32

train_count = len(train_image_path)
train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)

train_image_ds = train_image_ds.prefetch(AUTOTUNE)
test_image_path = glob.glob('../input/cat-and-dog/training_set/training_set/*/*.jpg')

test_image_label = [int(p.split('training_set/training_set')[1].split('/')[0] == 'cats') for p in test_image_path]

test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))

test_image_ds = test_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)

test_image_ds = test_image_ds.batch(BATCH_SIZE)

test_image_ds = test_image_ds.prefetch(AUTOTUNE)
len(test_image_path)
imgs,labels=next(iter(train_image_ds))
labels.shape
plt.imshow(imgs[0])
labels[0]
model = keras.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),

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

    tf.keras.layers.MaxPooling2D(),

     tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),

    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(1)

])
model.summary()
pred = model(imgs)
pred.shape
pred
np.array([p[0].numpy() for p in tf.cast(pred > 0, tf.int32)])
np.array([l[0].numpy() for l in labels])
ls = tf.keras.losses.BinaryCrossentropy()#定义损失函数
ls([0.,0.,1.,1.], [1.,1.,1.,1.])
ls([[0.],[0.],[1.],[1.]], [[1.],[1.],[1.],[1.]])
tf.keras.losses.binary_crossentropy([0.,0.,1.,1.], [1.,1.,1.,1.])
optimizer = tf.keras.optimizers.Adam()
epoch_loss_avg = tf.keras.metrics.Mean('train_loss')

train_accuracy = tf.keras.metrics.Accuracy()



epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')

test_accuracy = tf.keras.metrics.Accuracy()
train_accuracy([1,0,1], [1,1,1])
def train_step(model, images, labels):

    with tf.GradientTape() as t:

        pred = model(images)

        loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)

    grads = t.gradient(loss_step, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    epoch_loss_avg(loss_step)

    train_accuracy(labels, tf.cast(pred>0, tf.int32))
def test_step(model, images, labels):

    pred = model(images,training=False)

    loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)

    epoch_loss_avg_test(loss_step)

    test_accuracy(labels, tf.cast(pred>0, tf.int32))
train_loss_results = []

train_acc_results = []



test_loss_results = []

test_acc_results = []
num_epochs = 10
for epoch in range(num_epochs):

    for imgs_, labels_ in train_image_ds:

        train_step(model, imgs_, labels_)

        print('.', end='')

    print()

    

    train_loss_results.append(epoch_loss_avg.result())

    train_acc_results.append(train_accuracy.result())

    

    

    for imgs_, labels_ in test_image_ds:

        test_step(model, imgs_, labels_)

        

    test_loss_results.append(epoch_loss_avg_test.result())

    test_acc_results.append(test_accuracy.result())

    

    print('Epoch:{}: loss: {:.3f}, accuracy: {:.3f}, test_loss: {:.3f}, test_accuracy: {:.3f}'.format(

        epoch + 1,

        epoch_loss_avg.result(),

        train_accuracy.result(),

        epoch_loss_avg_test.result(),

        test_accuracy.result()

    ))

    

    epoch_loss_avg.reset_states()

    train_accuracy.reset_states()

    

    epoch_loss_avg_test.reset_states()

    test_accuracy.reset_states()