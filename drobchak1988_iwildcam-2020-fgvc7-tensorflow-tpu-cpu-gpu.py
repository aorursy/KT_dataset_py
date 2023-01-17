# https://github.com/visipedia/iwildcam_comp

# https://github.com/microsoft/CameraTraps/blob/master/megadetector.md
# !pip install gcsfs

from glob import glob

import math, os, time, re, json, shutil, pprint, random, gc

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets



print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE

# 

from tensorflow.keras.models import Model

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, MaxPooling2D, Activation, GlobalMaxPooling2D, concatenate

from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Input, SeparableConv2D, Flatten

from tensorflow.keras.layers import add as add_concat

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K



from IPython.display import clear_output

import IPython.display as display



import warnings

warnings.filterwarnings("ignore")

np.random.seed(0)
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
# TRAIN_PATERN = KaggleDatasets().get_gcs_path('iwidcam-2020-train-tfrecords')

# !gsutil ls $GCS_DS_PATH

TRAIN_PATERN = '/kaggle/input/iwidcam-2020-train-tfrecords'
if strategy.num_replicas_in_sync == 1: # single GPU or CPU

    BATCH_SIZE = 256

    VALIDATION_BATCH_SIZE = 256

else: # TPU pod

    BATCH_SIZE = 16 * strategy.num_replicas_in_sync

    VALIDATION_BATCH_SIZE = 16 * strategy.num_replicas_in_sync

    

FILENAMES = tf.io.gfile.glob(TRAIN_PATERN+'/*.tfrec')



IMAGE_SIZE = [64, 64]

IMAGE_TARGET = [64, 64]

if K.image_data_format() == 'channels_first':

    SHAPE = (3,*IMAGE_SIZE)

    INPUT_SHAPE = (3, *IMAGE_TARGET)

else:

    SHAPE = (*IMAGE_SIZE, 3)

    INPUT_SHAPE = (*IMAGE_TARGET, 3)

SIZE_TFRECORD = 1024

split = int(len(FILENAMES)*0.81)

TRAINING_FILENAMES = FILENAMES[:split]

VALIDATION_FILENAMES = FILENAMES[split:]

STEP_PER_EPOCH = (len(TRAINING_FILENAMES)*SIZE_TFRECORD)//BATCH_SIZE

VALIDATION_STEP_PER_EPOCH = (len(VALIDATION_FILENAMES)*SIZE_TFRECORD)//VALIDATION_BATCH_SIZE

print(len(TRAINING_FILENAMES))

print(len(VALIDATION_FILENAMES))
def read_tfrecord(example):

    features = {

        "image": tf.io.FixedLenFeature([], tf.string), 

        "class": tf.io.FixedLenFeature([], tf.int64), 

        "iage_id": tf.io.FixedLenFeature([], tf.string),

        "label": tf.io.VarLenFeature(tf.float32) ,

        "size": tf.io.FixedLenFeature([2], tf.int64) 

    }

    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)



    iage_id = example['iage_id']

    class_num = example['class']

    label = tf.sparse.to_dense(example['label'])



    return image,label



option_no_order = tf.data.Options()

option_no_order.experimental_deterministic = False
def display_9_images_from_dataset(dataset):

    plt.figure(figsize=(13,13))

    subplot=331

    for i, (image, label) in enumerate(dataset):

        plt.subplot(subplot)

        plt.axis('off')

        plt.imshow(image.numpy().astype(np.uint8))

        plt.title(label.numpy().decode("utf-8"), fontsize=16)

        subplot += 1

        if i==8:

            break

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))





def transform(image,label):

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size

    

    rot = 15. * tf.random.normal([1],dtype='float32')

    shr = 5. * tf.random.normal([1],dtype='float32') 

    h_zoom = 0.8 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 0.8 + tf.random.normal([1],dtype='float32')/10.

    h_shift = 16. * tf.random.normal([1],dtype='float32') 

    w_shift = 16. * tf.random.normal([1],dtype='float32') 

    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3]),label



def resize_and_crop_image(image, label):

    w = tf.shape(image)[0]

    h = tf.shape(image)[1]

    tw = IMAGE_TARGET[1]

    th = IMAGE_TARGET[0]

    resize_crit = (w * th) / (h * tw)

    image = tf.cond(resize_crit < 1,

                    lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true

                    lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false

                   )

    nw = tf.shape(image)[0]

    nh = tf.shape(image)[1]

    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)

    return image, label



def normalize(image, label):

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    image = tf.cast(image, tf.float32)/255.0 

    # image = tf.image.per_image_standardization(image)

    return image, label





def augmentation(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.rot90(image, k=random.randrange(4))

    image = tf.image.random_saturation(image, 0, 2)

    image = tf.image.random_hue(image, 0.08)

    image = tf.image.random_contrast(image, 0.7, 1.3)

    image = tf.image.random_brightness(image, 0.15)

    return image, label





def force_image_sizes(dataset):

    reshape_images = lambda image, label: (tf.reshape(image, SHAPE), label)   

    dataset = dataset.map(reshape_images, num_parallel_calls=AUTO)

    return dataset



def load_dataset(filenames):

    ignore_order = tf.data.Options()

    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)

    return dataset





def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES)

    dataset = dataset.map(transform, num_parallel_calls=AUTO)

    dataset = dataset.map(normalize, num_parallel_calls=AUTO)

#     dataset = force_image_sizes(dataset)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(30523)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(VALIDATION_FILENAMES)

    dataset = dataset.map(normalize, num_parallel_calls=AUTO)

#     dataset = force_image_sizes(dataset)

    dataset = dataset.batch(VALIDATION_BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) 

    return dataset
plt.ioff()

plt.rc('image', cmap='gray_r')

plt.rc('grid', linewidth=1)

plt.rc('xtick', top=False, bottom=False, labelsize='large')

plt.rc('ytick', left=False, right=False, labelsize='large')

plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')

plt.rc('text', color='a8151a')

plt.rc('figure', facecolor='F0F0F0', figsize=(16,9))

# Matplotlib fonts

MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")



def plot_learning_rate(lr_func, epochs):

    xx = np.arange(epochs+1, dtype=np.float)

    y = [lr_decay(x) for x in xx]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.set_xlabel('epochs')

    ax.set_title('Learning rate\ndecays from {:0.3g} to {:0.3g}'.format(y[0], y[-2]))

    ax.minorticks_on()

    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=1)

    ax.grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)

    ax.step(xx,y, linewidth=3, where='post')

    display.display(fig)



class PlotTraining(tf.keras.callbacks.Callback):

  def __init__(self, sample_rate=1, zoom=1):

    self.sample_rate = sample_rate

    self.step = 0

    self.zoom = zoom

    self.steps_per_epoch = STEP_PER_EPOCH



  def on_train_begin(self, logs={}):

    self.batch_history = {}

    self.batch_step = []

    self.epoch_history = {}

    self.epoch_step = []

    self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 7))

    plt.ioff()



  def on_batch_end(self, batch, logs={}):

    if (batch % self.sample_rate) == 0:

      self.batch_step.append(self.step)

      for k,v in logs.items():

        # do not log "batch" and "size" metrics that do not change

        # do not log training accuracy "acc"

        if k=='batch' or k=='size':# or k=='acc':

          continue

        self.batch_history.setdefault(k, []).append(v)

    self.step += 1



  def on_epoch_end(self, epoch, logs={}):

    plt.close(self.fig)

    self.axes[0].cla()

    self.axes[1].cla()

      

    self.axes[0].set_ylim(0, 1.2/self.zoom)

    self.axes[1].set_ylim(1-1/self.zoom/2, 1+0.1/self.zoom/2)

    

    self.epoch_step.append(self.step)

    for k,v in logs.items():

      # only log validation metrics

      if not k.startswith('val_'):

        continue

      self.epoch_history.setdefault(k, []).append(v)



    display.clear_output(wait=True)

    

    for k,v in self.batch_history.items():

      self.axes[0 if k.endswith('loss') else 1].plot(np.array(self.batch_step) / self.steps_per_epoch, v, label=k)

      

    for k,v in self.epoch_history.items():

      self.axes[0 if k.endswith('loss') else 1].plot(np.array(self.epoch_step) / self.steps_per_epoch, v, label=k, linewidth=3)

      

    self.axes[0].legend()

    self.axes[1].legend()

    self.axes[0].set_xlabel('epochs')

    self.axes[1].set_xlabel('epochs')

    self.axes[0].minorticks_on()

    self.axes[0].grid(True, which='major', axis='both', linestyle='-', linewidth=1)

    self.axes[0].grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)

    self.axes[1].minorticks_on()

    self.axes[1].grid(True, which='major', axis='both', linestyle='-', linewidth=1)

    self.axes[1].grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)

    display.display(self.fig)
with strategy.scope():

    bnmomemtum=0.88

    def fire(x, filters, kernel_size):

        if not isinstance(filters, list): 

            filters = [filters, filters]  

        x = SeparableConv2D(filters[0], kernel_size, padding='same', use_bias=False)(x)

        x = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)

        x = Activation('relu')(x)

        x = SeparableConv2D(filters[1], kernel_size, padding='same', use_bias=False)(x)

        return BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)

    

    def fire_module_separable_conv(filters, kernel_size=(3, 3)):

        return lambda x: fire(x, filters, kernel_size)

    

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    img_input = Input(shape=SHAPE)



    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)

    x = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)

    x = Activation('relu')(x)



    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)

    x = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)

    x = Activation('relu')(x)



    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)

    residual = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(residual)



    x = fire_module_separable_conv(128)(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = add_concat([x, residual])



    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)

    residual = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(residual)



    x = Activation('relu')(x)

    x = fire_module_separable_conv(256)(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = add_concat([x, residual])



    for i in range(4):

        residual = x



        x = Activation('relu')(x)

        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)

        x = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)

        x = Activation('relu')(x)

        x = fire_module_separable_conv(256)(x)

        

        x = add_concat([x, residual])





    x = fire_module_separable_conv([728, 1024])(x)

    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)



    x = Dense(572, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=[x]) 

    

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
if strategy.num_replicas_in_sync == 1: # single GPU

    start_lr =  0.0001

    min_lr = 0.000158

    max_lr = 0.01 * strategy.num_replicas_in_sync

    rampup_epochs = 14

    sustain_epochs = 0

    exp_decay = .7

else: # TPU pod

    start_lr =  0.0001

    min_lr = 0.000158

    max_lr = 0.01 * strategy.num_replicas_in_sync

    rampup_epochs = 14

    sustain_epochs = 0

    exp_decay = .7



EPOCHS=25



def lr_decay(epoch):

    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):

        if epoch < rampup_epochs:

            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr

        elif epoch < rampup_epochs + sustain_epochs:

            lr = max_lr

        else:

            lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr

        return lr

    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)

    

lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr_decay(epoch), verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lr_decay(x) for x in rng]

plt.plot(rng, [lr_decay(x) for x in rng])

print(y[0], y[-1])

plot_learning_rate(lr_decay_callback, EPOCHS)
weight_path="{}_weights.squeeze.hdf5".format('model')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True) 



plot_training = PlotTraining(sample_rate=10, zoom=1)

callbacks_list = [checkpoint, plot_training, lr_decay_callback] #, 
training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
history = model.fit(

    training_dataset, 

    steps_per_epoch=STEP_PER_EPOCH, 

    epochs=EPOCHS,

    validation_data=validation_dataset,

    validation_steps=VALIDATION_STEP_PER_EPOCH,

    callbacks=callbacks_list)
model.load_weights(weight_path)

model.save('model_tpu_Squeeze.h5')