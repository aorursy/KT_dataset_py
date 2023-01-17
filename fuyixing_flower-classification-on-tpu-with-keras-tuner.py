# Configuration

TESTING_LEVEL = 2



if TESTING_LEVEL == 0:

    # For debugging

    IMAGE_SIZE = [192, 192]

    EPOCHS_SEARCH = 5

    MAX_TRIALS = 3

    EPOCHS_FINAL = 5

    BATCH_SIZE_PER_REPLICA = 16

elif TESTING_LEVEL == 1:

    # For relatively short test to see some reasonable result

    IMAGE_SIZE = [331, 331]

    EPOCHS_SEARCH = 15

    MAX_TRIALS = 5

    EPOCHS_FINAL = 10

    BATCH_SIZE_PER_REPLICA = 16

else:

    # For an extended run.

    # Can set even larger MAX_TRIALS and EPOCHS_SEARCH for even better result.

    IMAGE_SIZE = [331, 331]

    EPOCHS_SEARCH = 60

    MAX_TRIALS = 10

    EPOCHS_FINAL = 10

    BATCH_SIZE_PER_REPLICA = 16
!pip install -q tensorflow==2.3.0 # Use 2.3.0 for built-in EfficientNet

!pip install -q git+https://github.com/keras-team/keras-tuner@master # Use github head for newly added TPU support

!pip install -q cloud-tpu-client # Needed for sync TPU version
import random, re, math

import numpy as np, pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf, tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets

print('Tensorflow version ' + tf.__version__)

import kerastuner as kt
# Detect hardware, return appropriate distribution strategy

try:

    # Sync TPU version

    from cloud_tpu_client import Client

    c = Client()

    c.configure_tpu_version(tf.__version__, restart_type='ifNeeded')

    

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None

    



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)

BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}



GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition
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
from tensorflow.data.experimental import AUTOTUNE



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) 

    # For keras.application implementation of EfficientNet, input should be [0, 255]

    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled = True, ordered = False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # Diregarding data order. Order does not matter since we will be shuffling the data anyway

    

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTOTUNE) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # use data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,

                          num_parallel_calls = AUTOTUNE) # returns a dataset of (image, label) pairs if labeled = True or (image, id) pair if labeld = False

    return dataset





def one_hot_encoding(image, label, num_classes=len(CLASSES)):

    return image, tf.one_hot(label, num_classes)



def get_training_dataset(dataset,do_aug=True):

    dataset = dataset.map(one_hot_encoding, num_parallel_calls=AUTOTUNE)

    if do_aug: dataset = dataset.map(transform, num_parallel_calls=AUTOTUNE)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # Drop remainder to ensure same batch size for all.

    dataset = dataset.prefetch(AUTOTUNE) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(dataset):

    dataset = dataset.map(one_hot_encoding, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTOTUNE) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)

    dataset = dataset.prefetch(AUTOTUNE) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

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

    image = tf.image.random_flip_left_right(image)

    

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    rot = 20. * tf.random.normal([1],dtype='float32')

    shr = 5. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

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
num_val_samples = count_data_items(VALIDATION_FILENAMES)

num_train_samples = count_data_items(TRAINING_FILENAMES)



train_ds = get_training_dataset(load_dataset(TRAINING_FILENAMES))

validation_ds = get_validation_dataset(load_dataset(VALIDATION_FILENAMES))

num_train_batches = num_train_samples // BATCH_SIZE

num_val_batches = num_val_samples // BATCH_SIZE
row = 3; col = 4;

all_elements = get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False).unbatch()

one_element = tf.data.Dataset.from_tensors( next(iter(all_elements)) )

augmented_element = one_element.repeat().map(transform).batch(row*col)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        plt.axis('off')

        plt.imshow(img[j,] / 255.)

    plt.show()

    break
import copy



# Helper function: re-compile with the same loss/metric/optimizer

def recompile(model):

    metrics = model.compiled_metrics.metrics

    metrics = [x.name for x in metrics]

    model.compile(loss=model.loss,

                  metrics=metrics,

                  optimizer=model.optimizer)



class FineTuner(kt.engine.tuner.Tuner):

    def run_trial(self, trial, *fit_args, **fit_kwargs):       

        copied_fit_kwargs = copy.copy(fit_kwargs)



        model = self.hypermodel.build(trial.hyperparameters)

        #dry run to build metrics

        model.evaluate(*fit_args, steps=1, batch_size=1)

        

        # freeze pretrained feature extractor

        for l in model.layers:

            # For efficientnet implementation we use, layers in the

            # Feature extraction part of model all have 'block', 

            # 'stem' or 'top_conv' in name.

            if any(x in l.name for x in ['block', 'stem', 'top_conv']):

                l.trainable = False

        recompile(model)

        model.fit(*fit_args, **copied_fit_kwargs)

        

        for l in model.layers:

            if not isinstance(l, tf.keras.layers.BatchNormalization):

                l.trainable = True

        recompile(model)

        

        

        callbacks = fit_kwargs.pop('callbacks', [])

        callbacks = self._deepcopy_callbacks(callbacks)

        # TunerCallback reports results to the `Oracle` and save the trained Model.

        callbacks.append(kt.engine.tuner_utils.TunerCallback(self, trial))

        copied_fit_kwargs['callbacks'] = callbacks

        model.fit(*fit_args, **copied_fit_kwargs)
# Define HyperModel using built-in application

from kerastuner.applications.efficientnet import HyperEfficientNet

hm = HyperEfficientNet(input_shape=[IMAGE_SIZE[0], IMAGE_SIZE[1], 3] , classes=len(CLASSES))



# Define Oracle

oracle = kt.tuners.bayesian.BayesianOptimizationOracle(

    objective='val_accuracy',

    max_trials=MAX_TRIALS,

)



# Initiate Tuner

tuner = FineTuner(

    hypermodel=hm,

    oracle=oracle,

    directory='flower_classification_kt_tpu',

    project_name='randomsearch_efficientnet',

    # Distribution strategy is passed in here.

    distribution_strategy=strategy,

    overwrite=True

)



tuner.search_space_summary()
tuner.search(train_ds,

             epochs=EPOCHS_SEARCH,

             validation_data=validation_ds,

             steps_per_epoch=num_train_batches,

             validation_steps=num_val_batches,

             # We can add callbacks here just like in model.fit()

             callbacks=[tf.keras.callbacks.ReduceLROnPlateau(),

                        tf.keras.callbacks.EarlyStopping(patience=5)],

             verbose=2)
tuner.results_summary()

model = tuner.get_best_models()[0]
ds_all =  get_training_dataset(load_dataset(TRAINING_FILENAMES + VALIDATION_FILENAMES))



# Train the best model with all data

model.fit(ds_all,

          epochs=EPOCHS_FINAL,

          steps_per_epoch=num_train_batches + num_val_batches,

          callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

          verbose=2)
ds_test = get_test_dataset(ordered=True)



print('Computing predictions...')

predictions = []



for i, (test_img, test_id) in enumerate(ds_test):

    print('Processing batch ', i)

    probabilities = model(test_img)

    prediction = np.argmax(probabilities, axis=-1)

    predictions.append(prediction)



predictions = np.concatenate(predictions)

print('Number of test examples predicted: ', predictions.shape)
# Get image ids from test set and convert to unicode

ds_test_ids = ds_test.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(ds_test_ids.batch(np.iinfo(np.int64).max))).numpy().astype('U')



# Write the submission file

np.savetxt(

    'submission.csv',

    np.rec.fromarrays([test_ids, predictions]),

    fmt=['%s', '%d'],

    delimiter=',',

    header='id,label',

    comments='',

)



# Look at the first few predictions

!head submission.csv