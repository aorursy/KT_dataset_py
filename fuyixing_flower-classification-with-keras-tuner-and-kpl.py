# Configuration

TESTING_LEVEL = 0



if TESTING_LEVEL == 0:

    # For debugging

    IMAGE_SIZE = [224, 224]

    EPOCHS_SEARCH = 5

    MAX_TRIALS = 3

    EPOCHS_FINAL = 50

    BATCH_SIZE_PER_REPLICA = 16

elif TESTING_LEVEL == 1:

    # For relatively short test to see some reasonable result

    IMAGE_SIZE = [224, 224]

    EPOCHS_SEARCH = 10

    MAX_TRIALS = 5

    EPOCHS_FINAL = 10

    BATCH_SIZE_PER_REPLICA = 16

else:

    # For an extended run.

    # Can set even larger MAX_TRIALS and EPOCHS_SEARCH for even better result.

    IMAGE_SIZE = [224, 224]

    EPOCHS_SEARCH = 30

    MAX_TRIALS = 10

    EPOCHS_FINAL = 10

    BATCH_SIZE_PER_REPLICA = 32

    



# load previous keras tuner result from output if possible

!cp -r ../input/flower-kt-record/flower_classification_kt_kpl .
!pip install -q tensorflow==2.3.0

!pip install -q git+https://github.com/keras-team/keras-tuner@master
import random, re, math

import numpy as np, pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf, tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets

print('Tensorflow version ' + tf.__version__)

import kerastuner as kt

tf.config.optimizer.set_jit(True)
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

print(GCS_DS_PATH)



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

    # For keras.application.efficientnet, inputs range [0, 255]

    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3])

    return image



def read_labeled_tfrecord(example, num_classes=len(CLASSES)):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "class": tf.io.FixedLenFeature([], tf.int64),

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    label = tf.one_hot(label, num_classes)

    return image, label



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "id": tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords.

    # When ordering is not needed, set `ordered=False` for faster loading.

    

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False 



    # automatically interleaves reads from multiple files

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)

    # use data as soon as it streams in, rather than in its original order

    dataset = dataset.with_options(ignore_order)

    if labeled:

        dataset = dataset.map(read_labeled_tfrecord,

                              num_parallel_calls=AUTOTUNE)

    else:

        dataset = dataset.map(read_unlabeled_tfrecord,

                              num_parallel_calls=AUTOTUNE)

    return dataset





def get_training_dataset(filenames):

    dataset = load_dataset(filenames, labeled=True, ordered=False)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    # Drop remainder to ensure same batch size for all.

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Prefetch next batch while training

    dataset = dataset.prefetch(AUTOTUNE) 

    return dataset



def get_validation_dataset(filenames):

    dataset = load_dataset(filenames, labeled=True, ordered=False)

    # Drop remainder to ensure same batch size for all.

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Prefer not prefetching for validation data on GPUs.

    # dataset = dataset.prefetch(AUTOTUNE)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
num_val_samples = count_data_items(VALIDATION_FILENAMES)

num_train_samples = count_data_items(TRAINING_FILENAMES)



num_train_batches = num_train_samples // BATCH_SIZE

num_val_batches = num_val_samples // BATCH_SIZE



train_ds = get_training_dataset(TRAINING_FILENAMES)

validation_ds = get_validation_dataset(VALIDATION_FILENAMES)

all_ds = get_training_dataset(TRAINING_FILENAMES + VALIDATION_FILENAMES)
# function for visualizing augmentation model

def visualize(aug_model):

    row = 3; col = 4;

    element = train_ds.unbatch().take(1)



    for (img, label) in element:

        img_batch = tf.repeat(tf.expand_dims(img, 0),row * col,axis=0)

        img_augment = aug_model(img_batch)

        plt.figure(figsize=(15, int(15 * row / col)))

        for j in range(row * col):

            plt.subplot(row,col, j + 1)

            plt.axis('off')

            plt.imshow(img_augment[j,] / 255.)

        plt.show()

        break
from kerastuner.applications import HyperImageAugment



hm_aug = HyperImageAugment(

    input_shape=[*IMAGE_SIZE, 3],

    augment_layers=0,

    rotate=[0.3, 0.5], # range of factor of rotation

    translate_x=None, # horizontal translation is off

    translate_y=[0.3, 0.5], # range of factor of vertical translation

    contrast=None) # auto contrast is off
from kerastuner.engine.hyperparameters import HyperParameters

hp = HyperParameters()

aug_model = hm_aug.build(hp)

aug_model.summary()



visualize(aug_model)
from kerastuner.applications import HyperImageAugment



hm_aug = HyperImageAugment(

    input_shape=[*IMAGE_SIZE, 3],

    augment_layers=[1, 1], # only one layer of augmentation

    rotate=[0.3, 0.5], # range of factor of rotation

    translate_x=None, # horizontal translation is off

    translate_y=[0.3, 0.5], # range of factor of vertical translation

    contrast=None) # auto contrast is off



hp = HyperParameters()

aug_model = hm_aug.build(hp)

aug_model.summary()



visualize(aug_model)
from kerastuner.applications import HyperImageAugment

# Use default setting

hm_aug = HyperImageAugment(input_shape=[*IMAGE_SIZE, 3]) 
# Define HyperModel using built-in application

from kerastuner.applications.efficientnet import HyperEfficientNet



hm = HyperEfficientNet(

    input_shape=[*IMAGE_SIZE, 3],

    classes=len(CLASSES),

    # Augmentation model goes here. It can be HyperModel or Keras Model.

    augmentation_model=hm_aug) 



# Optional: Restrict default hyperparameters.

# To take effect, pass this `hp` instance when constructing tuner as `hyperparameters=hp`

from kerastuner.engine.hyperparameters import HyperParameters

hp = HyperParameters()

# Restrict choice of EfficientNet version from B0-B7 to B0-B3

hp.Choice('version', ['B0', 'B1', 'B2', 'B3']) 

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

        callbacks = fit_kwargs.pop('callbacks', [])

        callbacks = self._deepcopy_callbacks(callbacks)

        copied_fit_kwargs['callbacks'] = callbacks

        

        tf.keras.backend.clear_session()

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

            if isinstance(l, tf.keras.layers.BatchNormalization):

                l.trainable = True

        # it usually suggested to increase learning rate if running with

        # multiple replica because of the increased batch size.

        model.optimizer.lr = model.optimizer.lr * strategy.num_replicas_in_sync

        recompile(model)

        model.fit(*fit_args, **copied_fit_kwargs)

        

        for l in model.layers:

            l.trainable = True

        model.optimizer.lr = model.optimizer.lr / 10

        recompile(model)

        

        # TunerCallback reports results to the `Oracle` and save the trained Model.

        callbacks.append(kt.engine.tuner_utils.TunerCallback(self, trial))

        

        model.fit(*fit_args, **copied_fit_kwargs)
# Define Oracle

oracle = kt.tuners.bayesian.BayesianOptimizationOracle(

    objective='val_accuracy',

    max_trials=MAX_TRIALS,

    hyperparameters=hp,

)



# Initiate Tuner

tuner = FineTuner(

    hypermodel=hm,

    oracle=oracle,

    directory='flower_classification_kt_kpl',

    project_name='bayesian_efficientnet',

    # Distribution strategy is passed in here.

    distribution_strategy=strategy, 

    # optimizer can be set here.

    optimizer='adam',

    metrics=['accuracy'],

    )

tuner.search_space_summary()
# tuner.search(train_ds,

#              epochs=EPOCHS_SEARCH,

#              validation_data=validation_ds,

#              steps_per_epoch=num_train_batches,

#              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(),

#                         tf.keras.callbacks.EarlyStopping(patience=5)],

#              verbose=2)
tuner.results_summary()

model = tuner.get_best_models()[0]
# Train the best model with all data

model.fit(all_ds,

          epochs=EPOCHS_FINAL,

          steps_per_epoch=num_train_batches + num_val_batches,

          callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss')],

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