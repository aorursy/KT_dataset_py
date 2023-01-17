!pip install -q tensorflow-gpu==2.0.0-beta1
!pip install -q tensorflow-datasets
%load_ext autoreload

%autoreload 2
batch_size = 32

shuffle_buffer = 1000
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import logging



logging.getLogger('tensorflow').setLevel(logging.ERROR)



from tensorflow.keras.callbacks import Callback



class CosineAnnealer:

    

    def __init__(self, start, end, steps):

        self.start = start

        self.end = end

        self.steps = steps

        self.n = 0

        

    def step(self):

        self.n += 1

        cos = np.cos(np.pi * (self.n / self.steps)) + 1

        return self.end + (self.start - self.end) / 2. * cos





class OneCycleScheduler(Callback):

    """`Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).

    If the model supports a momentum parameter, it will also be adapted by the schedule.

    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where

    only two phases are used and the adaptation is done using cosine annealing.

    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.

    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.

    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.

    """



    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):

        super(OneCycleScheduler, self).__init__()

        lr_min = lr_max / div_factor

        final_lr = lr_max / (div_factor * 1e4)

        phase_1_steps = steps * phase_1_pct

        phase_2_steps = steps - phase_1_steps

        

        self.phase_1_steps = phase_1_steps

        self.phase_2_steps = phase_2_steps

        self.phase = 0

        self.step = 0

        

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 

                 [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

        

        self.lrs = []

        self.moms = []



    def on_train_begin(self, logs=None):

        self.phase = 0

        self.step = 0



        self.set_lr(self.lr_schedule().start)

        self.set_momentum(self.mom_schedule().start)

        

    def on_train_batch_begin(self, batch, logs=None):

        self.lrs.append(self.get_lr())

        self.moms.append(self.get_momentum())



    def on_train_batch_end(self, batch, logs=None):

        self.step += 1

        if self.step >= self.phase_1_steps:

            self.phase = 1

            

        self.set_lr(self.lr_schedule().step())

        self.set_momentum(self.mom_schedule().step())

        

    def get_lr(self):

        try:

            return tf.keras.backend.get_value(self.model.optimizer.lr)

        except AttributeError:

            return None

        

    def get_momentum(self):

        try:

            return tf.keras.backend.get_value(self.model.optimizer.momentum)

        except AttributeError:

            return None

        

    def set_lr(self, lr):

        try:

            tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        except AttributeError:

            pass # ignore

        

    def set_momentum(self, mom):

        try:

            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)

        except AttributeError:

            pass # ignore



    def lr_schedule(self):

        return self.phases[self.phase][0]

    

    def mom_schedule(self):

        return self.phases[self.phase][1]

    

    def plot(self):

        ax = plt.subplot(1, 2, 1)

        ax.plot(self.lrs)

        ax.set_title('Learning Rate')

        ax = plt.subplot(1, 2, 2)

        ax.plot(self.moms)

        ax.set_title('Momentum')

        
class LRFinder(Callback):

    """`Callback` that exponentially adjusts the learning rate after each training batch between `start_lr` and

    `end_lr` for a maximum number of batches: `max_step`. The loss and learning rate are recorded at each step allowing

    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via

    the `plot` method.

    """



    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 1000, smoothing=0.9):

        super(LRFinder, self).__init__()

        self.start_lr, self.end_lr = start_lr, end_lr

        self.max_steps = max_steps

        self.smoothing = smoothing

        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0

        self.lrs, self.losses = [], []



    def on_train_begin(self, logs=None):

        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0

        self.lrs, self.losses = [], []



    def on_train_batch_begin(self, batch, logs=None):

        self.lr = self.exp_annealing(self.step)

        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)



    def on_train_batch_end(self, batch, logs=None):

        logs = logs or {}

        loss = logs.get('loss')

        step = self.step

        if loss:

            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss

            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))

            self.losses.append(smooth_loss)

            self.lrs.append(self.lr)



            if step == 0 or loss < self.best_loss:

                self.best_loss = loss



            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):

                self.model.stop_training = True



        if step == self.max_steps:

            self.model.stop_training = True



        self.step += 1



    def exp_annealing(self, step):

        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)



    def plot(self):

        fig, ax = plt.subplots(1, 1)

        ax.set_ylabel('Loss')

        ax.set_xlabel('Learning Rate (log scale)')

        ax.set_xscale('log')

        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

        ax.plot(self.lrs, self.losses)
fashion_mnist = tf.keras.datasets.fashion_mnist



(x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()

x_train, x_valid = x_train / 255.0, x_valid / 255.0



x_train = x_train[..., tf.newaxis]

x_valid = x_valid[..., tf.newaxis]



train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(shuffle_buffer).batch(batch_size)

valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, Dropout



def build_model():

    # LeNet-5 CNN

    return tf.keras.models.Sequential([

        Conv2D(6, 3, padding='same', activation='relu'),

        AveragePooling2D(),

        Conv2D(16, 3, padding='valid', activation='relu'),

        AveragePooling2D(),

        Flatten(),

        Dense(120, activation='relu'),

        Dense(84, activation='relu'),

        Dense(10, activation='softmax')

    ])
lr_finder = LRFinder()



model = build_model()

optimizer = tf.keras.optimizers.RMSprop()

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

_ = model.fit(train_ds, epochs=10, callbacks=[lr_finder], verbose=True)



lr_finder.plot()
epochs = 3

lr = 5e-3

steps = np.ceil(len(x_train) / batch_size) * epochs

lr_schedule = OneCycleScheduler(lr, steps)



model = build_model()

optimizer = tf.keras.optimizers.RMSprop(lr=lr)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=valid_ds, epochs=epochs, callbacks=[lr_schedule], verbose=True)
lr_schedule.plot()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
import tensorflow_datasets as tfds

tfds.disable_progress_bar()



dataset_splits = (8, 2)

splits = tfds.Split.TRAIN.subsplit(weighted=dataset_splits)



(raw_train, raw_validation), metadata = tfds.load('cats_vs_dogs', split=list(splits), with_info=True, as_supervised=True)
img_size = 160



def resize_image(image, label):

    image = tf.cast(image, tf.float32)

    image = (image/127.5) - 1

    image = tf.image.resize(image, (img_size, img_size))

    return image, label
train = raw_train.map(resize_image).batch(batch_size)

validation = raw_validation.map(resize_image).batch(batch_size)
train_batches = train.shuffle(shuffle_buffer)

validation_batches = validation
img_shape = (img_size, img_size, 3)
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense



def build_pretrained_model():

    base_model = MobileNetV2(input_shape=img_shape, weights='imagenet', include_top=False)

    base_model.trainable=False



    return tf.keras.Sequential([

        base_model,

        GlobalAveragePooling2D(),

        Dense(1)

    ])
lr_finder = LRFinder(max_steps=500)

model = build_pretrained_model()

optimizer = tf.keras.optimizers.Adam(amsgrad=True)

model.compile(optimizer=optimizer, loss='binary_crossentropy')

try:

    model.fit(train_batches, epochs=20, callbacks=[lr_finder], verbose=True)

except:

    pass

lr_finder.plot()
num_train, num_val = (

  metadata.splits['train'].num_examples*split/10

  for split in dataset_splits

)

epochs = 5

lr = 5e-3

steps = np.ceil(num_train / batch_size) * epochs

lr_schedule = OneCycleScheduler(lr, steps)



model = build_pretrained_model()

optimizer = tf.keras.optimizers.RMSprop(lr=lr)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_batches, validation_data=validation_batches, epochs=epochs, callbacks=[lr_schedule], verbose=True)