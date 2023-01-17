!pip install -U git+https://github.com/qubvel/efficientnet >> /dev/null
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

import tensorflow.keras.backend as K



import efficientnet.tfkeras as efn



from sklearn.metrics import *



from kaggle_datasets import KaggleDatasets



import gc

import time

import re

import math



AUTO = tf.data.experimental.AUTOTUNE
# Detect TPU, return appropriate distribution strategy

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
MIXED_PRECISION = True

XLA_ACCELERATE = True



if MIXED_PRECISION:

    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')

    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

    mixed_precision.set_policy(policy)

    print('Mixed precision enabled')



if XLA_ACCELERATE:

    tf.config.optimizer.set_jit(True)

    print('Accelerated Linear Algebra enabled')
# TODO : Try with bigger image size

# TODO : Try with larger batch sizes



IMAGE_SIZES = [[512, 512], [331, 331], [224, 224], [192, 192]]

IMAGE_SIZE = IMAGE_SIZES[2]



BATCH_SIZE = 32 * strategy.num_replicas_in_sync



print('Using IMAGE_SIZE', IMAGE_SIZE, 'and BATCH_SIZE', BATCH_SIZE)
use_external = True



BASE_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')

EXT_PATH = KaggleDatasets().get_gcs_path('tf-flower-photo-tfrec')



def get_path(image_size=IMAGE_SIZE):

    # image_size is [512, 512] for example

    return f'{BASE_PATH}/tfrecords-jpeg-{image_size[0]}x{image_size[1]}'



def get_ext_path(folder, image_size=IMAGE_SIZE):

    return f'{EXT_PATH}/{folder}/tfrecords-jpeg-{image_size[0]}x{image_size[1]}'



TRAINING_FILES = tf.io.gfile.glob(get_path() + '/train/*.tfrec')

VALIDATION_FILES = tf.io.gfile.glob(get_path() + '/val/*.tfrec')

TEST_FILES = tf.io.gfile.glob(get_path() + '/test/*.tfrec')



if use_external:

#     IMAGENET = tf.io.gfile.glob(get_ext_path('imagenet_no_test') + '/*.tfrec')

#     TRAINGING_FILES += IMAGENET

    INATURALIST = tf.io.gfile.glob(get_ext_path('inaturalist_no_test') + '/*.tfrec')

    TRAINING_FILES += INATURALIST

#     OPENIMAGE = tf.io.gfile.glob(get_ext_path('openimage_no_test') + '/*.tfrec')

#     TRAINING_FILES += OPENIMAGE

    OXFORD = tf.io.gfile.glob(get_ext_path('oxford_102_no_test') + '/*.tfrec')

    TRAINING_FILES += OXFORD

    TFFLOWERS = tf.io.gfile.glob(get_ext_path('tf_flowers_no_test') + '/*.tfrec')

    TRAINING_FILES += TFFLOWERS



print('Loaded Training:', len(TRAINING_FILES), 'files, Validation:', len(VALIDATION_FILES), 'files, Test:', len(TEST_FILES), 'files')
CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102



print(len(CLASSES))
# TODO : Find other augmentations that can be used with TPU with TF



class Augmentor():

    

    @staticmethod

    def process(img, size):

        img = Augmentor.transform(img, IMAGE_SIZE[0])

        img = tf.image.random_brightness(img, 0.3)

        img = tf.image.random_contrast(img, 0.8, 1.2)

        img = tf.image.random_flip_left_right(img)

        img = tf.image.random_flip_up_down(img)

#         img = tf.image.random_hue(img, 0.2)

        img = tf.image.random_saturation(img, 0.8, 1.2)

        img = Augmentor.dropout(img, size)

        return img



    @staticmethod

    def dropout(img, size, probability=0.5, count=2, ratio=0.4):

        probability = tf.cast( 

            tf.random.uniform([],0,1) < probability, 

            tf.int32

        )

        if (probability == 0) | (count == 0) | (ratio == 0):

            return img



        for k in range(count):

            x = tf.cast( tf.random.uniform([], 0, size), tf.int32)

            y = tf.cast( tf.random.uniform([], 0, size), tf.int32)

            

            # COMPUTE SQUARE 

            width = tf.cast( ratio*size, tf.int32 ) * probability

            ya = tf.math.maximum(0, y-width//2)

            yb = tf.math.minimum(size, y+width//2)

            xa = tf.math.maximum(0,x-width//2)

            xb = tf.math.minimum(size, x+width//2)

            

            # DROPOUT IMAGE

            one = img[ya:yb, 0:xa, :]

            two = tf.random.normal([yb-ya, xb-xa, 3])

            three = img[ya:yb, xb:size, :]

            middle = tf.concat([one,two,three],axis=1)

            img = tf.concat([img[0:ya,:,:], middle, img[yb:size,:,:]], axis=0)



        # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR 

        img = tf.reshape(img, [size, size, 3])

        return img



    @staticmethod

    def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

        # returns 3x3 transformmatrix which transforms indicies



        # CONVERT DEGREES TO RADIANS

        rotation = math.pi * rotation / 180.

        shear    = math.pi * shear    / 180.



        def get_3x3_mat(lst):

            return tf.reshape(tf.concat([lst],axis=0), [3,3])



        # ROTATION MATRIX

        c1   = tf.math.cos(rotation)

        s1   = tf.math.sin(rotation)

        one  = tf.constant([1],dtype='float32')

        zero = tf.constant([0],dtype='float32')



        rotation_matrix = get_3x3_mat([c1,   s1,   zero, 

                                       -s1,  c1,   zero, 

                                       zero, zero, one])    

        # SHEAR MATRIX

        c2 = tf.math.cos(shear)

        s2 = tf.math.sin(shear)    



        shear_matrix = get_3x3_mat([one,  s2,   zero, 

                                    zero, c2,   zero, 

                                    zero, zero, one])        

        # ZOOM MATRIX

        zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 

                                   zero,            one/width_zoom, zero, 

                                   zero,            zero,           one])    

        # SHIFT MATRIX

        shift_matrix = get_3x3_mat([one,  zero, height_shift, 

                                    zero, one,  width_shift, 

                                    zero, zero, one])



        return K.dot(K.dot(rotation_matrix, shear_matrix), 

                     K.dot(zoom_matrix,     shift_matrix))





    @staticmethod

    def transform(image, DIM=256):    

        # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

        # output - image randomly rotated, sheared, zoomed, and shifted

        XDIM = DIM%2 #fix for size 331



        rot = 180.0 * tf.random.normal([1], dtype='float32')

        shr = 2.0 * tf.random.normal([1], dtype='float32') 

        h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 8.0

        w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 8.0

        h_shift = 8.0 * tf.random.normal([1], dtype='float32') 

        w_shift = 8.0 * tf.random.normal([1], dtype='float32') 



        # GET TRANSFORMATION MATRIX

        m = Augmentor.get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



        # LIST DESTINATION PIXEL INDICES

        x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)

        y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])

        z   = tf.ones([DIM*DIM], dtype='int32')

        idx = tf.stack( [x,y,z] )



        # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

        idx2 = K.dot(m, tf.cast(idx, dtype='float32'))

        idx2 = K.cast(idx2, dtype='int32')

        idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)



        # FIND ORIGIN PIXEL VALUES           

        idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])

        d    = tf.gather_nd(image, tf.transpose(idx3))



        return tf.reshape(d,[DIM, DIM,3])
class DS():

    

    def __init__(self, files, training=True, shuffle=False, augment=False, batch_size=16, repeat=False):

        self.files = files

        self.augment = augment



        self.ds = tf.data.TFRecordDataset(files, num_parallel_reads = AUTO)

        self.ds = self.ds.map(self.read_with_label if training else self.read_without_label)

        

        if self.augment:

            self.ds = self.ds.map(self.augment_image)

        

        if shuffle:

            options = tf.data.Options()

            options.experimental_deterministic = False

            self.ds = self.ds.with_options(options)

            self.ds = self.ds.shuffle(2048)

            

        if repeat:

            self.ds = self.ds.repeat()



        self.ds = self.ds.batch(batch_size)

        self.ds = self.ds.prefetch(AUTO) # Improves throughput

        # self.ds = self.ds.cache() # Uses too much memory, ignore



    def data(self):

        return self.ds

        

    def read_with_label(self, example):

        example = tf.io.parse_single_example(example, {

            'image': tf.io.FixedLenFeature([], tf.string),

            'class': tf.io.FixedLenFeature([], tf.int64),

        })

        return self.decode_image(example['image']), tf.cast(example['class'], tf.int32)

    

    def read_without_label(self, example):

        example = tf.io.parse_single_example(example, {

            'image': tf.io.FixedLenFeature([], tf.string),

            'id': tf.io.FixedLenFeature([], tf.string),

        })

        return self.decode_image(example['image']), example['id']

    

    def decode_image(self, img):

        img = tf.io.decode_jpeg(img, channels=3)

        img = tf.cast(img, tf.float32) / 255.0

        return tf.reshape(img, [*IMAGE_SIZE, 3])

    

    def augment_image(self, img, label):

        return Augmentor.process(img, IMAGE_SIZE[0]), label

    

    def count(self):

        # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in self.files]

        return np.sum(n)

items = 0

batches = 0

times = { 

    'batches': list(),

    'items': list(),

    'batch_speed': list(),

    'item_speed': list(),

}



def loop(items, batches, times):

    train_ds = DS(TRAINING_FILES, augment=True, batch_size=128).data()

#     train_ds = DS(VALIDATION_FILES, batch_size=128).data()



    start = time.time()

    for element in train_ds:

        so_far = time.time() - start

        batches += 1

        items += len(element[0])



        times['batches'].append(batches)

        times['items'].append(items)

        times['batch_speed'].append((batches+1)/so_far)

        times['item_speed'].append((items+1)/so_far)



        print(items, so_far, (items)/so_far)



    print(batches, items, so_far, (batches)/so_far, (items)/so_far)

    gc.collect()



def show_graph(size):

    plt.figure(figsize=(12,6))

    plt.plot(times['batches'], times['batch_speed'], color='red')

    plt.plot(times['items'], times['item_speed'], color='blue', linestyle='--')

    plt.legend()

    plt.title('Batch size' + str(size))

    plt.show()

    

# loop(items, batches, times) # 270-280 images/sec, 170-180/sec with albumenations, 60 with tensorflow

# show_graph(128)
def test_dataset():

    print('Training images')

    show_images(DS(TRAINING_FILES, augment=True, batch_size=128).data())

    print('Validation image')

    show_images(DS(VALIDATION_FILES, batch_size=128).data())

    print('Testing images')

    show_images(DS(TEST_FILES, training=False, batch_size=128).data())

    print('Testing Augmented images')

    show_images(DS(TEST_FILES, training=False, augment=True, batch_size=128).data())

    

def show_images(ds, rows=3, columns=6):

    plt.figure(figsize=(12, 6))

    count = 0

    

    for i, examples in enumerate(ds.take(1)):

        images = examples[0]

        labels = examples[1]

        for j, image in enumerate(images):

            if count == rows * columns:

                break

            plt.subplot(rows, columns, count+1, xticks=[], yticks=[])

            plt.imshow(image)

            if labels[0].dtype == tf.string:

                plt.xlabel(labels[j].numpy().decode("utf-8"))

            else:

                plt.xlabel(CLASSES[labels[j]])

            count += 1

    plt.tight_layout()

    plt.show()

    

# test_dataset()

gc.collect()
EFNS = [

    efn.EfficientNetB0, 

    efn.EfficientNetB1, 

    efn.EfficientNetB2, 

    efn.EfficientNetB3,

    efn.EfficientNetB4,

    efn.EfficientNetB5,

    efn.EfficientNetB6,

    efn.EfficientNetB7,

]



def create_efficientnet(index, weight='imagenet'):

    pretrained_model = EFNS[index](

        weights = weight,

        include_top = False,

        input_shape = [*IMAGE_SIZE, 3]

    )

    pretrained_model.trainable = True

    

    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax'),

    ], name='efn' + str(index))

    return model



def create_densenet():

    pretrained_model = tf.keras.applications.DenseNet201(

        weights = 'imagenet',

        include_top = False,

        input_shape = [*IMAGE_SIZE, 3]

    )

    pretrained_model.trainable = True

    

    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax'),

    ], name='densenet')

    return model



def create_cnn():

    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=[*IMAGE_SIZE, 3]),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ], name='simple')

    return model
# TODO : Make a model class



def compile_models(models):

    for x in models:

        models[x].compile(

            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),

            loss = tf.keras.losses.SparseCategoricalCrossentropy(),

            metrics = ['sparse_categorical_accuracy'],

        )

        models[x].summary()

        print('')



models = {}



with strategy.scope():

    

#     start = time.time()

#     models['simple'] = create_cnn()

#     print(f'Created simple in {(time.time() - start):.4f}s')

    

#     start = time.time()

#     models['simple2'] = create_cnn()

#     print(f'Created simple2 in {(time.time() - start):.4f}s')



    start = time.time()

    models['efn7'] = create_efficientnet(7)

    print(f'Created efn7 in {(time.time() - start):.4f}s')



    start = time.time()

    models['efn6'] = create_efficientnet(6)

    print(f'Created efn6 in {(time.time() - start):.4f}s')

    

#     start = time.time()

#     models['efn5'] = create_efficientnet(5)

#     print(f'Created efn5 in {(time.time() - start):.4f}s')



#     start = time.time()

#     models['efn4'] = create_efficientnet(4)

#     print(f'Created efn4 in {(time.time() - start):.4f}s')



    start = time.time()

    models['efn4-noisy'] = create_efficientnet(4, 'noisy-student')

    print(f'Created efn4 in {(time.time() - start):.4f}s')



#     start = time.time()

#     models['efn3'] = create_efficientnet(3)

#     print(f'Created efn3 in {(time.time() - start):.4f}s')



    start = time.time()

    models['densenet'] = create_densenet()

    print(f'Created densenet in {(time.time() - start):.4f}s')

    



compile_models(models)
train_ds = DS(TRAINING_FILES, augment=True, shuffle=True, batch_size=BATCH_SIZE, repeat=True)

val_ds = DS(VALIDATION_FILES, batch_size=BATCH_SIZE)

val_aug_ds = DS(VALIDATION_FILES, augment=True, batch_size=BATCH_SIZE)

test_ds = DS(TEST_FILES, training=False, batch_size=BATCH_SIZE)



print(f'Training: {train_ds.count()} images, Validation: {val_ds.count()}, Test: {test_ds.count()} images')
# TODO : Try with CosineDecayRestarts in LearningRateScheduler

# Didn't work : change lr_max to use 

#    LR_START = 0.00001

#    LR_MAX = 0.00005 * strategy.num_replicas_in_sync

#    LR_MIN = 0.00001



#1 

lr_start = 0.00005

lr_max = 0.0000125 * strategy.num_replicas_in_sync #BATCH_SIZE

lr_min = 0.0000001

lr_ramp_ep = 5

lr_sus_ep = 0

lr_decay = 0.8



def lrfn(epoch):

    if epoch < lr_ramp_ep:

        lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

    elif epoch < lr_ramp_ep + lr_sus_ep:

        lr = lr_max

    else:

        lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

    return lr



lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



count = range(100)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)

plt.plot(count, [lrfn(x) for x in count])



#2

cosine_decay_learning_rate = tf.keras.experimental.CosineDecayRestarts(

        0.001, 0.8, t_mul=2.0, m_mul=1.0, alpha=0.0,

        name=None

    )



# lr_callback = tf.keras.callbacks.LearningRateScheduler(cosine_decay_learning_rate, verbose=True)
class Visibility(tf.keras.callbacks.Callback):

    

    def __init__(self):

        super(Visibility, self).__init__()

        self.metrics = ['sparse_categorical_accuracy', 'loss']

    

    def on_train_begin(self, logs=None):

        # Training starts here

        self.train_begin = time.time()



    def on_train_end(self, logs=None):

        print('\n Training took ', time.time() - self.train_begin, 'seconds')



#     def on_epoch_begin(self, epoch, logs=None):

#         pass



    def on_epoch_end(self, epoch, logs=None):

        output = ""

        for metric in self.metrics:

            diff = ((logs['val_'+metric] / logs[metric]) - 1)*100

            output += f"val/{metric}: {diff:.2f}% "

        print("\n" + output)



    def on_train_batch_begin(self, batch, logs=None):

        print("▪", end="")



#     def on_test_begin(self, logs=None):

#         # Validation starts

#         pass

#     def on_test_end(self, logs=None):

#         pass

#     def on_train_batch_end(self, batch, logs=None):

#         pass

#     def on_test_batch_begin(self, batch, logs=None):

#         pass



    def on_test_batch_end(self, batch, logs=None):

        print("▫", end="")



#     def on_predict_begin(self, logs=None):

#         pass

#     def on_predict_end(self, logs=None):

#         pass

#     def on_predict_batch_begin(self, batch, logs=None):

#         pass

#     def on_predict_batch_end(self, batch, logs=None):

#         pass
EPOCHS = 50

history = {}



def fit_models(models, history):

    for x in models:

        print(f'Starting training for {x}')

        hist = models[x].fit(

            train_ds.data(),

            validation_data = val_ds.data(),

            epochs = EPOCHS,

            steps_per_epoch = train_ds.count() // BATCH_SIZE,

            verbose=2,

            callbacks=[

                tf.keras.callbacks.ModelCheckpoint(

                    monitor='val_sparse_categorical_accuracy', verbose=2, save_best_only=True, save_weights_only=True, mode='max', filepath=f'{x}.h5'),

                lr_callback,

                tf.keras.callbacks.EarlyStopping(

                    monitor='val_sparse_categorical_accuracy', min_delta=0.001, patience=3, verbose=2, mode='max', baseline=None, restore_best_weights=True

                ),

                Visibility(),

            ],

        )

        history[x] = hist



fit_models(models, history)
def plot_metrics(name, h):

    plt.figure(figsize=(14, 6))



    plt.subplot(1,2,1)

    plt.plot(h.epoch, h.history['loss'], color='blue')

    plt.plot(h.epoch, h.history['val_loss'], color='blue', linestyle='--')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.title(name)



    plt.subplot(1,2,2)

    plt.plot(h.epoch, h.history['sparse_categorical_accuracy'], color='orange')

    plt.plot(h.epoch, h.history['val_sparse_categorical_accuracy'], color='orange', linestyle='--')

    plt.xlabel('Epochs')

    plt.ylabel('Sparse Categorical Accuracy')

    plt.legend(loc='upper right')

    plt.title(name)



    plt.show()

    

def plot_overfitting(name, h):

    loss_gap = np.array(h.history['val_loss']) - np.array(h.history['loss'])

    acc_gap = 100*(np.array(h.history['sparse_categorical_accuracy']) 

                   - np.array(h.history['val_sparse_categorical_accuracy']))



    plt.figure(figsize=(14, 6))



    plt.subplot(1, 3, 1)

    plt.xlabel('Epochs')

    plt.ylabel('Loss Gap')

    plt.plot(h.epoch, loss_gap, color='blue')

    plt.title(name)



    plt.subplot(1, 3, 2)

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy Gap')

    plt.plot(h.epoch, acc_gap, color='orange')

    plt.title(name)



    plt.subplot(1, 3, 3)

    plt.xlabel('Epochs')

    plt.ylabel('Learning Rate')

    plt.plot(h.epoch, h.history['lr'], color='red')

    plt.title(name)



    plt.legend()

    plt.show()

    

for x in history:

    plot_metrics(x, history[x])

    plot_overfitting(x, history[x])
highest_f1 = 0

highest_f1_model = ''

preds = list()



def get_y_true():



    def get_val_classes():

        all_classes = np.array([])

        for examples in val_ds.data():

            all_classes = np.concatenate([all_classes, examples[1].numpy()])

        return all_classes



    y_true = get_val_classes()

    print('True Values:: Length:', len(y_true), ', List:', y_true)

    return y_true



def predict_models(models, y_true):

    global highest_f1

    global highest_f1_model

    global preds

    

     # TPU doesn't like strings, get rid of them

    val_tpu_ds = val_aug_ds.data().map(lambda image, ids: image)

    for x in models:

        val_aug_count = 2

        y_val_pred = np.zeros((3712,104), dtype=np.float64)

        for i in range(val_aug_count):

            y_val_loop = models[x].predict(val_tpu_ds, verbose=1)

            y_val_loop = tf.cast(y_val_loop, tf.float32)

            y_val_pred += y_val_loop.numpy()

        y_val_pred /= val_aug_count

        y_val = np.argmax(y_val_pred, axis=-1).astype(np.float32)



        print(f'\n\nScores for {x}\n\n')

        print(f'{x}: Pred Values:: Length:{len(y_val)} List: {y_val}')

        

        model_f1_score = f1_score(y_true, y_val, average='macro', zero_division=0)*100

        if model_f1_score > highest_f1:

            highest_f1 = model_f1_score

            highest_f1_model = x



        scores = {

            "metrics": [

                "F1",

                "Precision",

                "Recall",

            ],

            "scores": [

                model_f1_score,

                precision_score(y_true, y_val, average='macro', zero_division=0)*100,

                recall_score(y_true, y_val, average='macro', zero_division=0)*100,

            ]

        }

        

        print(pd.DataFrame.from_dict(scores))



        preds.append((

            x,

            y_val_pred,

            model_f1_score,

        ))

        # print( classification_report(y_true, y_val, target_names=CLASSES) )



y_true = get_y_true()

predict_models(models, y_true)
# x, 1

# y, alpha1

# z, alpha2



def get_best_alpha(pred1, pred2):

    # Given two predictions, find the highest alpha and f1

    best_alpha = 0

    best_f1 = 0

    

    for x in np.linspace(0, 1, 101):

        mixed = (pred1 * x)  + (pred2 * (1-x))

        y_mixed = np.argmax(mixed, axis=-1).astype(np.float32)

        mixed_f1 = f1_score(y_true, y_mixed, average='macro', zero_division=0)*100        

        if mixed_f1 > best_f1:

            best_f1 = mixed_f1

            best_alpha = x

    return best_alpha, best_f1





def get_alpha(preds, y_true):

    output = list()

    

    # sort the model by highest f1 score

    preds = np.array(preds)

    preds = preds[preds[:,2].argsort()[::-1]]

    

    model_name = preds[0][0]

    predictions = preds[0][1]

    model_f1 = preds[0][2]

    output.append((model_name, 1))

    

    best_predictions = predictions

    best_f1 = model_f1

    print(best_f1, output)



    for i in range(1, len(preds)):

        i_name = preds[i][0]

        i_preds = preds[i][1]

        i_f1 = preds[i][1]

        

        alpha, f1 = get_best_alpha(best_predictions, i_preds)

        if f1 > best_f1:

            best_predictions = (alpha * i_preds) + ((1-alpha) * best_predictions)

            best_f1 = f1

            output.append((i_name, alpha))

            print(best_f1, output)

            

    return output

    

alpha_chain = get_alpha(preds, y_true)
print(f'{highest_f1_model} has the highest f1 score - {highest_f1:0.5f}')
# TODO : Create an ensemble from multiple models



ds_submission = DS(TEST_FILES, training=False, augment=True, batch_size=BATCH_SIZE).data()

ds_submission = ds_submission.map(lambda image, ids: image)



def get_test_ids():

    all_ids = np.array([])

    for examples in test_ds.data():

        all_ids = np.concatenate([all_ids, examples[1].numpy().astype('U')])

    return all_ids



def get_prediction_with_tta(model, ds_submission):

    

    tta = 5

    

    y_pred = np.zeros((7382,104), dtype=np.float64)

    

    for x in range(tta):    

        y_pred_loop = model.predict(ds_submission, verbose=2)

        y_pred_loop = tf.cast(y_pred_loop, tf.float32)

        y_pred += y_pred_loop.numpy()

    

    y_pred /= tta

    return y_pred

    

def create_submission(alpha_chain, models, ds_submission):

    

    y_pred = np.zeros((7382,104), dtype=np.float64)

    

    for x in alpha_chain:

        model = x[0]

        alpha = x[1]

        

        print('Predicting with', model, 'with alpha', alpha)

        y_pred_chain = get_prediction_with_tta(models[model], ds_submission)

        y_pred = (alpha * y_pred_chain) + ((1-alpha) * y_pred)

    

    y_pred = np.argmax(y_pred, axis=-1)



    filename = f'submission.csv'

    np.savetxt(

        filename, 

        np.column_stack((get_test_ids(), y_pred)),

        fmt=('%s', '%s'),

        delimiter=',',

        header='id,label',

        comments=''

    )



# create_submission(models[highest_f1_model])

create_submission(alpha_chain, models, ds_submission)