!pip install tensorflow_datasets jax jaxlib flax
import jax

from jax import random

import jax.numpy as jnp



import flax

from flax import nn

from flax import optim



import tensorflow_datasets as tfds

import tensorflow.compat.v2 as tf

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from keras.utils import to_categorical

tfds.disable_progress_bar()

tf.enable_v2_behavior()



import time
device_name = tf.test.gpu_device_name()

if "GPU" not in device_name:

    print("GPU not found")

else:

    print('Found GPU at: {}'.format(device_name))

    gpus = tf.config.experimental.list_physical_devices('GPU')

    tf.config.experimental.set_memory_growth(gpus[0], True)
train_ds = tfds.load('cifar10', split=tfds.Split.TRAIN)

train_ds = train_ds.map(lambda x: {'image': tf.cast(x['image'], tf.float32) / 255.,

                                     'label': tf.cast(x['label'], tf.int32)})

train_ds = train_ds.cache().shuffle(1000)

tmp_train = train_ds.batch(16)

train_ds = train_ds.batch(128)

test_ds = tfds.as_numpy(tfds.load(

      'cifar10', split=tfds.Split.TEST, batch_size=-1))

test_ds = {'image': test_ds['image'].astype(jnp.float32) / 255.,

             'label': test_ds['label'].astype(jnp.int32)}
mini_batch = next(tfds.as_numpy(tmp_train))
#I genuinely have no idea how exactly dropout must be implemented

#in this framework as devs themselves admit the documentation on

#stochastic context manager is 'a bit sparse'

class CNN(nn.Module):

    def apply(self, x):

        x = nn.Conv(x, features=96, kernel_size=(3, 3))

        x = nn.relu(x)

        x = nn.Conv(x, features=96, kernel_size=(3, 3), strides = (2,2))

        x = nn.relu(x)

        with nn.stochastic(random.PRNGKey(0)): 

            x = nn.dropout(x, rate = 0.2)

        x = nn.Conv(x, features=192, kernel_size=(3, 3))

        x = nn.relu(x)

        x = nn.Conv(x, features=96, kernel_size=(3, 3), strides = (2,2))

        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.BatchNorm(x, use_running_average=not train_step, momentum = 0.99, epsilon=1e-3, name='init_bn')

        x = nn.Dense(x, features=256)

        x = nn.relu(x)

        x = nn.Dense(x, features=10)

        x = nn.log_softmax(x)

        return x
def onehot(labels, num_classes=10):

    return (labels[..., None] == jnp.arange(num_classes)[None]).astype(jnp.float32)



def cross_entropy_loss(preds, labels):

    return -jnp.mean(jnp.sum(onehot(labels) * preds, axis=-1))

#We could also implement it for single element and vectorize later with vmap



def compute_metrics(preds, labels):

    return {'loss': cross_entropy_loss(preds, labels),

            'accuracy': jnp.mean(jnp.argmax(preds, -1) == labels)}
#Using jit decorator for GPU acceleration for entire function

@jax.jit

def train_step(optimizer, batch):

    def loss_fn(model):

        preds = model(batch['image'])

        loss = cross_entropy_loss(preds, batch['label'])

        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (_, preds), grad = grad_fn(optimizer.target)

    optimizer = optimizer.apply_gradient(grad)

    return optimizer



@jax.jit

def eval_step(model, batch):

    preds = model(batch['image'])

    return compute_metrics(preds, batch['label'])



def eval_model(model, test_ds):

    metrics = eval_step(model, test_ds)

    metrics = jax.device_get(metrics)

    summary = jax.tree_map(lambda x: x.item(), metrics)

    return summary['loss'], summary['accuracy']
learning_rate = 0.001

beta = 0.9

beta_2 = 0.999
_, init_params = CNN.init_by_shape(random.PRNGKey(0), [((1, 32, 32, 3), jnp.float32)])

model = nn.Model(CNN, init_params)

optimizer = optim.Adam(learning_rate=learning_rate, beta1=beta, beta2 = beta_2).create(model)
#jit pre-compilation    

start_mini = time.monotonic()

train_step(optimizer, mini_batch)

mini_time = time.monotonic() - start_mini

print('mini_batch training: %.2fs' % mini_time)

    

start_mini_2 = time.monotonic()

eval_model(optimizer.target, test_ds)

mini_val_time = time.monotonic() - start_mini_2

print('mini_batch validation: %.2fs' % mini_val_time)
#Mini-batch after compilation

start_mini = time.monotonic()

train_step(optimizer, mini_batch)

mini_time = time.monotonic() - start_mini

print('mini_batch training: %.2fs' % mini_time)

    

start_mini_2 = time.monotonic()

eval_model(optimizer.target, test_ds)

mini_val_time = time.monotonic() - start_mini_2

print('mini_batch validation: %.2fs' % mini_val_time)
for epoch in range(1, 2):

    batch_gen = tfds.as_numpy(train_ds)

    for batch in batch_gen:

        optimizer = train_step(optimizer, batch)
def train(train_ds, test_ds, model, optimizer):



    batch_size = 128

    num_epochs = 10

    learning_rate = 0.001

    beta = 0.9

    beta_2 = 0.999

    loss = 0

    accuracy = 0

    

    start_time = time.monotonic()

    

    for epoch in range(1, num_epochs + 1):

        train_time = 0

        start_time_3 = time.monotonic()

        batch_gen = tfds.as_numpy(train_ds)

        for batch in batch_gen:

            start_time_step = time.monotonic()

            optimizer = train_step(optimizer, batch)

            train_time += time.monotonic() - start_time_step

            

        flax_step = time.monotonic() - start_time_3

        

        start_time_2 = time.monotonic()

        loss, accuracy = eval_model(optimizer.target, test_ds)

        flax_inf = time.monotonic() - start_time_2

        

        print('eval epoch: %d, epoch: %.2fs, actual_training: %.2fs, validation: %.2fs, loss: %.4f, accuracy: %.2f' % 

              (epoch, flax_step, train_time, flax_inf, loss, accuracy * 100))

        

    flax_time = time.monotonic() - start_time

    return optimizer, flax_time, accuracy, flax_inf
_, flax_time, flax_acc, flax_inf = train(train_ds, test_ds, model, optimizer)
#Slightly different data loading pipeline

(ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True,)



def normalize_img(image, label):

    return tf.cast(image, tf.float32) / 255., label



ds_train = ds_train.map(normalize_img)

ds_train = ds_train.cache()

ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)

ds_train = ds_train.batch(128)

ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)



ds_test = ds_test.map(normalize_img)

ds_test = ds_test.batch(128)

ds_test = ds_test.cache()

ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
model = Sequential([

    Conv2D(input_shape=(32,32,3), filters=96, kernel_size=(3,3), activation='relu'),

    Conv2D(filters=96, kernel_size=(3,3), strides=2, activation='relu'),

    Dropout(0.2),

    Conv2D(filters=192, kernel_size=(3,3), activation='relu'),

    Conv2D(filters=192, kernel_size=(3,3), strides=2, activation='relu'),

    Flatten(),

    BatchNormalization(),

    Dense(256, activation='relu'),

    Dense(10, activation='softmax')

])



model.compile(

    loss='sparse_categorical_crossentropy',

    optimizer=Adam(0.001),

    metrics=['accuracy'],

)
start_time = time.monotonic()



model.fit(

    ds_train,

    epochs=10,

    validation_data=ds_test)



tf_time = time.monotonic() - start_time
start_time = time.monotonic()



loss, tf_acc = model.evaluate(ds_test)



tf_inf = time.monotonic() - start_time
from keras import backend as K 

K.clear_session()



!pip install numba

from numba import cuda

cuda.select_device(0)

cuda.close()
from prettytable import PrettyTable

t = PrettyTable(['', 'Flax', 'TensorFlow'])

t.add_row(['Train time', '%.2fs'%(flax_time), '%.2fs'%(tf_time)])

t.add_row(['Inference time', '%.2fs'%(flax_inf), '%.2fs'%(tf_inf)])

t.add_row(['Accuracy', '%.2f%%'%(flax_acc * 100), '%.2f%%'%(tf_acc * 100)])

print(t)