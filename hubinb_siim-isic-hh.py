colab=0

show_files=0

tstamp=0

if colab:

    from google.colab import drive

    drive.mount('/content/gdrive')

if (not colab)&show_files:

    import os

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))
!pip install -q efficientnet

import math

import pytz

import random

import numpy as np

import pandas as pd

import math, re, os, gc

import tensorflow as tf

from pathlib import Path

from datetime import datetime

from scipy.stats import rankdata

import efficientnet.tfkeras as efn

from matplotlib import pyplot as plt

from sklearn.utils import class_weight

from sklearn.metrics import roc_auc_score



print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE



if not colab:

    from kaggle_datasets import KaggleDatasets
def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)

NAME='EffNB0_512'

NFOLDS=5

NBEST=2 # the number of best models to use for predictions

SEED=311



if colab:

    PATH=Path('/content/gdrive/My Drive/kaggle/input/siim-isic-melanoma-classification/') 

    train=pd.read_csv(PATH/'train.csv.zip')

else:

    PATH=Path('/kaggle/input/siim-isic-melanoma-classification/')

    train=pd.read_csv(PATH/'train.csv')



test=pd.read_csv(PATH/'test.csv')

sub=pd.read_csv(PATH/'sample_submission.csv')



seed_everything(SEED)

print(f"The shape of the training set is {train.shape}.")

print(f"The shape of the testing set is {test.shape}.")
print(f"The columns in `train`:\n {list(train.columns)}.\n")

print(f"The columns in `test`:\n {list(test.columns)}.")
train.head()

test.head()
try:

    # TPU detection. No parameters necessary if TPU_NAME environment 

    # variable is set. On Kaggle this is always the case.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_PATH={}



if colab:

    GCS_PATH['train']='gs://kds-4169ce1e5a624d4b5bd866480682658f2063b18c5cf995cd3316a7e9'

    GCS_PATH['test']='gs://kds-147ff1400c195e2f8c7b01492e04fd2769e0305e9ef6c01afc9c5870'

else:

    GCS_PATH['train']=KaggleDatasets().get_gcs_path('siim-512x512-tfrec-q95')

    GCS_PATH['test']=KaggleDatasets().get_gcs_path('siim-512x512-tfrec-q95-test')



print(GCS_PATH['train'])

print(GCS_PATH['test'])
%%time



IMAGE_SIZE = [512, 512] # At this size, a GPU will run out of memory. Use the TPU.

                          # For GPU training, please select 224 x 224 px image size.

EPOCHS=20

BATCH_SIZE = 8 * strategy.num_replicas_in_sync



CLASSES = ['benign', 'malignant']
ALL_TRAIN=tf.io.gfile.glob(GCS_PATH['train'] + '/*.tfrec')



VAL_FNAMES={}

for fn in range(1, NFOLDS+1):

    VAL_FNAMES[f"fold_{fn}"]=[path for path in ALL_TRAIN if f"fold_{fn}" in path]    

    print("Fold", f'{fn}:', len(VAL_FNAMES[f'fold_{fn}']), "elements in total.")

    

TRAIN_FNAMES={f'fold_{i}': list(set(ALL_TRAIN)-set(VAL_FNAMES[f'fold_{i}']))

              for i in range(1, NFOLDS+1)}



TEST_FNAMES = tf.io.gfile.glob(GCS_PATH['test'] + '/*.tfrec')
len(ALL_TRAIN), len(TEST_FNAMES), len(TRAIN_FNAMES), len(VAL_FNAMES)
VAL_FNAMES

TRAIN_FNAMES

ALL_TRAIN
def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, 

    # i.e. test10-687.tfrec = 687 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    

    return np.sum(n)
%%time



N_TRAIN_IMGS = {f'fold_{i}': count_data_items(TRAIN_FNAMES[f'fold_{i}'])

                for i in range(1, NFOLDS+1)}



N_VAL_IMGS = {f'fold_{i}': count_data_items(VAL_FNAMES[f'fold_{i}'])

              for i in range(1, NFOLDS+1)}



N_TEST_IMGS = count_data_items(TEST_FNAMES)



STEPS_PER_EPOCH = {f'fold_{i}': N_TRAIN_IMGS[f'fold_{i}'] // BATCH_SIZE

                   for i in range(1, NFOLDS+1)}



print("="*75)



print(f"The number of unlabeled test image is {N_TEST_IMGS}. It is common for all folds.")



for i in range(1, NFOLDS+1):

    print("="*75)

    print(f"Fold {i}: {N_TRAIN_IMGS[f'fold_{i}']} training and {N_VAL_IMGS[f'fold_{i}']} validation images.")

print("="*75)
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    # convert image to floats in [0, 1] range

    image = tf.cast(image, tf.float32) / 255.0 

    # explicit size needed for TPU

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    

    return image
def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        # tf.string means bytestring

        # shape [] means single element

        ################################

        # bytestring features

        "image": tf.io.FixedLenFeature([], tf.string), 

        "image_name": tf.io.FixedLenFeature([], tf.string),

        "patient_id": tf.io.FixedLenFeature([], tf.string),

        "benign_malignant": tf.io.FixedLenFeature([], tf.string),

        # integer features

        "age": tf.io.FixedLenFeature([], tf.int64),

        "sex_female": tf.io.FixedLenFeature([], tf.int64),        

        "sex_male": tf.io.FixedLenFeature([], tf.int64),

        "sex_unknown": tf.io.FixedLenFeature([], tf.int64),

        "site_head/neck": tf.io.FixedLenFeature([], tf.int64),

        "site_lower extremity": tf.io.FixedLenFeature([], tf.int64),

        "site_oral/genital": tf.io.FixedLenFeature([], tf.int64),

        "site_palms/soles": tf.io.FixedLenFeature([], tf.int64),

        "site_torso": tf.io.FixedLenFeature([], tf.int64),

        "site_unknown": tf.io.FixedLenFeature([], tf.int64),

        "site_upper extremity": tf.io.FixedLenFeature([], tf.int64),

        "height": tf.io.FixedLenFeature([], tf.int64),

        "width": tf.io.FixedLenFeature([], tf.int64),

        "target": tf.io.FixedLenFeature([], tf.int64), 

        # float features

        "age_scaled": tf.io.FixedLenFeature([], tf.float32),

    }



    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    # image data

    image = decode_image(example['image']) 

    data={}

    # bytestring features

    data['image_name']=image_name=tf.cast(example['image_name'], tf.string)

    data['patient_id']=tf.cast(example['patient_id'], tf.string)

    # integer features

    data['age']=tf.cast(example['age'], tf.int32)

    data['sex_female']=tf.cast(example['sex_female'], tf.int32)

    data['sex_male']=tf.cast(example['sex_male'], tf.int32)

    data['sex_unknown']=tf.cast(example['sex_unknown'], tf.int32)

    data['site_head/neck']=tf.cast(example['site_head/neck'], tf.int32)

    data['site_lower extremity']=tf.cast(example['site_lower extremity'], tf.int32)

    data['site_oral/genital']=tf.cast(example['site_oral/genital'], tf.int32)

    data['site_palms/soles']=tf.cast(example['site_palms/soles'], tf.int32)

    data['site_torso']=tf.cast(example['site_torso'], tf.int32)

    data['site_unknown']=tf.cast(example['site_unknown'], tf.int32)

    data['site_upper extremity']=tf.cast(example['site_upper extremity'], tf.int32)

#     data['height']=tf.cast(example['height'], tf.int32)

#     data['width']=tf.cast(example['width'], tf.int32)

    # float features

    data['age_scaled']=tf.cast(example['age_scaled'], tf.float32)

    # target (integer)

    label=tf.cast(example['target'], tf.int32)

     # target (string)

    label_name=tf.cast(example['benign_malignant'], tf.string)



    return image, label, data, label_name

def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        # tf.string means bytestring

        # shape [] means single element

        ################################

        # bytestring features

        "image": tf.io.FixedLenFeature([], tf.string), 

        "image_name": tf.io.FixedLenFeature([], tf.string),

        "patient_id": tf.io.FixedLenFeature([], tf.string),

        # integer features

        "age": tf.io.FixedLenFeature([], tf.int64),

        "sex_female": tf.io.FixedLenFeature([], tf.int64),        

        "sex_male": tf.io.FixedLenFeature([], tf.int64),

        "sex_unknown": tf.io.FixedLenFeature([], tf.int64),

        "site_head/neck": tf.io.FixedLenFeature([], tf.int64),

        "site_lower extremity": tf.io.FixedLenFeature([], tf.int64),

        "site_oral/genital": tf.io.FixedLenFeature([], tf.int64),

        "site_palms/soles": tf.io.FixedLenFeature([], tf.int64),

        "site_torso": tf.io.FixedLenFeature([], tf.int64),

        "site_unknown": tf.io.FixedLenFeature([], tf.int64),

        "site_upper extremity": tf.io.FixedLenFeature([], tf.int64),

        "height": tf.io.FixedLenFeature([], tf.int64),

        "width": tf.io.FixedLenFeature([], tf.int64), 

        # float features

        "age_scaled": tf.io.FixedLenFeature([], tf.float32),

    }



    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    # image data

    image = decode_image(example['image']) 

    data={}

    # bytestring features

    data['image_name']=image_name=tf.cast(example['image_name'], tf.string)

    data['patient_id']=tf.cast(example['patient_id'], tf.string)

    # integer features

    data['age']=tf.cast(example['age'], tf.int32)

    data['sex_female']=tf.cast(example['sex_female'], tf.int32)

    data['sex_male']=tf.cast(example['sex_male'], tf.int32)

    data['sex_unknown']=tf.cast(example['sex_unknown'], tf.int32)

    data['site_head/neck']=tf.cast(example['site_head/neck'], tf.int32)

    data['site_lower extremity']=tf.cast(example['site_lower extremity'], tf.int32)

    data['site_oral/genital']=tf.cast(example['site_oral/genital'], tf.int32)

    data['site_palms/soles']=tf.cast(example['site_palms/soles'], tf.int32)

    data['site_torso']=tf.cast(example['site_torso'], tf.int32)

    data['site_unknown']=tf.cast(example['site_unknown'], tf.int32)

    data['site_upper extremity']=tf.cast(example['site_upper extremity'], tf.int32)

#     data['height']=tf.cast(example['height'], tf.int32)

#     data['width']=tf.cast(example['width'], tf.int32)

    # float features

    data['age_scaled']=tf.cast(example['age_scaled'], tf.float32)



    return image, data

def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files 

    # at once and disregarding data order. Order does not matter since we will 

    # be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        # disable order, increase speed

        ignore_order.experimental_deterministic = False



    # automatically interleaves reads from multiple files

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.with_options(ignore_order)

    # returns a dataset of (image, label) pairs if labeled=True 

    # or (image, id) pairs if labeled=False

    dataset = dataset.map(read_labeled_tfrecord if labeled 

                          else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    

    return dataset
%%time



training_dataset = load_dataset(TRAIN_FNAMES['fold_2'])



print("Example of the training data:")

for image, label, data, label_name in training_dataset.take(1):

    print("The image batch size:", image.numpy().shape)

    print("Label:", label.numpy())

    print("Label name:", label_name.numpy())

    print("Age:", data['age'].numpy())

    print("Age (scaled):", data['age_scaled'].numpy())
%%time



validation_dataset = load_dataset(VAL_FNAMES['fold_2'])



print("Examples of the validation data:")

for image, label, data, label_name in validation_dataset.take(1):

    print("The image batch size:", image.numpy().shape)

    print("Label:", label.numpy())

    print("Label name:", label_name.numpy())

    print("Age:", data['age'].numpy())

    print("Age (scaled):", data['age_scaled'].numpy())
def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FNAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO)

    

    return dataset
%%time



test_dataset = get_test_dataset()



print("Examples of the test data:")

for image, data in test_dataset.take(1):

    print("The image batch size:", image.numpy().shape)

    print("Ages, 5 examples:", data['age'].numpy()[:5])

    print("Age (scaled), 5 examples:", data['age_scaled'].numpy()[:5])
np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(databatch):

    if len(databatch)==4:

        images, labels, _, _ = databatch

        numpy_images = images.numpy()

        numpy_labels = labels.numpy()

    else:

        images, _ = databatch

        numpy_images = images.numpy()

        numpy_labels = [None for _ in enumerate(numpy_images)]



    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" 

                                if not correct else '', 

                                CLASSES[correct_label] if not correct else ''), correct

def display_one_image(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), 

                  color='red' if red else 'black', fontdict={'verticalalignment':'center'}, 

                  pad=int(titlesize/1.5)

                 )

    return (subplot[0], subplot[1], subplot[2]+1)

def display_batch_of_images(databatch, predictions=None):

    """This will work with:

    display_batch_of_images(images)

    display_batch_of_images(images, predictions)

    display_batch_of_images((images, labels))

    display_batch_of_images((images, labels), predictions)

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does  

    # not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else CLASSES[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        # magic formula tested to work from 1x1 to 10x10 images

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3

        subplot = display_one_image(image, title, subplot, 

                                     not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()
training_dataset = training_dataset.batch(20)

train_batch = iter(training_dataset)

# run this cell again for next set of images

display_batch_of_images(next(train_batch))
del training_dataset, train_batch

gc.collect()
validation_dataset = validation_dataset.batch(20)

validation_batch = iter(validation_dataset)

# run this cell again for next set of images

display_batch_of_images(next(validation_batch))
del validation_dataset, validation_batch

gc.collect()
test_dataset = test_dataset.unbatch().batch(20)

test_batch = iter(test_dataset)

# run this cell again for next set of images

display_batch_of_images(next(test_batch))
del test_dataset, test_batch

gc.collect()
LR_START = 0.000005#0.00001

LR_MAX = 0.00000725 * strategy.num_replicas_in_sync

LR_MIN = 0.000005

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 4

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



# def lrfn(epoch):

#     LR_START = 0.00001

#     LR_MAX = 0.00005

#     LR_MIN = 0.00001

#     LR_RAMPUP_EPOCHS = 5

#     LR_SUSTAIN_EPOCHS = 0

#     LR_EXP_DECAY = .8

    

#     if epoch < LR_RAMPUP_EPOCHS:

#         lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

#     elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

#         lr = LR_MAX

#     else:

#         lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

#     return lr



lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
train['target'].values

class_weights = class_weight.compute_class_weight(class_weight='balanced',

                                                  classes=np.unique(train['target'].values),

                                                  y=train['target'].values,

                                                 )



class_weights = {i : class_weights[i] for i in range(len(class_weights))}



print(class_weights)
tab_feats=['age_scaled',

           'sex_female', 

           'sex_male', 

           'sex_unknown', 

           'site_head/neck', 

           'site_lower extremity', 

           'site_oral/genital',

           'site_palms/soles',

           'site_torso',

           'site_unknown',

           'site_upper extremity',

#            'height',

#            'width',

          ]



N_TAB_FEATS=len(tab_feats)



print(f"The number of tabular features is {N_TAB_FEATS}.")
%time



def get_model():

    with strategy.scope():

        pretrained_model = efn.EfficientNetB7(input_shape=(*IMAGE_SIZE, 3),

                                              weights='imagenet',

                                              include_top=False

                                             )

        # False = transfer learning, True = fine-tuning

        pretrained_model.trainable = True#False 



        inp1 = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')

        inp2 = tf.keras.layers.Input(shape=(N_TAB_FEATS), name='inp2')

        

        # BUILD MODEL HERE

        

        x=pretrained_model(inp1)

        x=tf.keras.layers.GlobalAveragePooling2D()(x)

        x=tf.keras.layers.Dense(256, 

                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

                                activation='relu')(x)

        x=tf.keras.layers.Dense(512, 

                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

                                activation='relu')(x)

        x=tf.keras.layers.Dense(1024, 

                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

                                activation='relu')(x)

        x=tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

                                activation='relu')(x)

        x=tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

                                activation='relu')(x)

        y=tf.keras.layers.Dense(100, 

                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

                                activation='relu')(inp2)

        

        concat=tf.keras.layers.concatenate([y, x])

        

        output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(concat)

        

        model = tf.keras.models.Model(inputs=[inp1,inp2], outputs=[output])

    

        model.compile(

        optimizer='adam',

        loss = 'binary_crossentropy',

        metrics=[tf.keras.metrics.AUC()],

        )

        

        return model
# %time



# def get_model():

#     with strategy.scope():

#         pretrained_model = efn.EfficientNetB7(input_shape=(*IMAGE_SIZE, 3),

#                                               weights='imagenet',

#                                               include_top=False

#                                              )

#         # False = transfer learning, True = fine-tuning

#         pretrained_model.trainable = True#False 



#         inp1 = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')

#         inp2 = tf.keras.layers.Input(shape=(N_TAB_FEATS), name='inp2')

        

#         # BUILD MODEL HERE

        

#         x=pretrained_model(inp1)

#         x=tf.keras.layers.GlobalAveragePooling2D()(x)

#         x=tf.keras.layers.Dense(512, 

#                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

#                                 activation='relu')(x)

#         x=tf.keras.layers.Dropout(0.2)(x)

#         x=tf.keras.layers.Dense(256, 

#                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

#                                 activation='relu')(x)

#         x=tf.keras.layers.Dropout(0.2)(x)

#         x=tf.keras.layers.Dense(128, 

#                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

#                                 activation='relu')(x)

#         x=tf.keras.layers.Dropout(0.2)(x)

#         x=tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

#                                 activation='relu')(x)

#         x=tf.keras.layers.Dropout(0.2)(x)

        

#         y=tf.keras.layers.Dense(100, 

#                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),

#                                 activation='relu')(inp2)

        

#         concat=tf.keras.layers.concatenate([y, x])

        

#         output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(concat)

        

#         model = tf.keras.models.Model(inputs=[inp1,inp2], outputs=[output])

    

#         model.compile(

#         optimizer='adam',

#         loss = 'binary_crossentropy',

#         metrics=[tf.keras.metrics.AUC()],

#         )

        

#         return model
%%time

model=get_model()

model.summary()
del model

gc.collect()
if colab:

    

    SAVE_FOLDER=NAME

    

    if tstamp:

        time_zone = pytz.timezone('America/Chicago')

        current_datetime = datetime.now(time_zone)

        ts=current_datetime.strftime("%m%d%H%M%S")

        SAVE_FOLDER+='_'+ts

        

    SAVE_FOLDER=PATH/SAVE_FOLDER

    if not os.path.exists(SAVE_FOLDER):

        os.mkdir(SAVE_FOLDER)



else:

    SAVE_FOLDER=Path('/kaggle/working')

class save_best_n(tf.keras.callbacks.Callback):

    def __init__(self, fn, model):

        self.fn = fn

        self.model = model



    def on_epoch_end(self, epoch, logs=None):

        

        if (epoch>0):

            score=logs.get("val_auc")

        else:

            score=-1

      

        if (score > best_score[fold_num].min()):

          

            idx_min=np.argmin(best_score[fold_num])



            best_score[fold_num][idx_min]=score

            best_epoch[fold_num][idx_min]=epoch+1



            path_best_model=f'best_model_fold_{self.fn}_{idx_min}.hdf5'

            self.model.save(SAVE_FOLDER/path_best_model)
def setup_input(image, label, data, label_name):

    

    tab_data=[tf.cast(data[tfeat], dtype=tf.float32) for tfeat in tab_feats]

    

    tabular=tf.stack(tab_data)

    

    return {'inp1': image, 'inp2':  tabular}, label

def data_augment(data, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement 

    # in the next function (below), this happens essentially for free on TPU. 

    # Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    data['inp1'] = tf.image.random_flip_left_right(data['inp1'])

    data['inp1'] = tf.image.random_flip_up_down(data['inp1'])

    #image = tf.image.random_saturation(image, 0, 2)

    

    return data, label

def get_training_dataset(dataset):

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    # the training dataset must repeat for several epochs

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    #dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)

    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO)

    

    return dataset

def get_validation_dataset(dataset):

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO)

    

    return dataset
%%time



debug=1

histories = []



best_epoch={fn: np.zeros(NBEST) for fn in range(1, NFOLDS+1)}

best_score={fn: np.zeros(NBEST) for fn in range(1, NFOLDS+1)}



for fold_num in range(1, NFOLDS+1):

    

    tf.keras.backend.clear_session()

    # clear tpu memory (otherwise can run into Resource Exhausted Error)

    # see https://www.kaggle.com/c/flower-classification-with-tpus/discussion/131045

    tf.tpu.experimental.initialize_tpu_system(tpu)

    

    print("="*50)

    print(f"Starting fold {fold_num} out of {NFOLDS}...")

    

    files_trn=TRAIN_FNAMES[f"fold_{fold_num}"]

    files_val=VAL_FNAMES[f"fold_{fold_num}"]

    

    if debug:

        files_trn=files_trn[0:2]

        files_val=files_val[0:2]

        EPOCHS=1

       

    train_dataset = load_dataset(files_trn)

    train_dataset = train_dataset.map(setup_input, num_parallel_calls=AUTO)

    

    val_dataset = load_dataset(files_val, ordered = True)

    val_dataset = val_dataset.map(setup_input, num_parallel_calls=AUTO)

    

    model = get_model()

    

    STEPS_PER_EPOCH = count_data_items(files_trn) // BATCH_SIZE

    

    print(f'STEPS_PER_EPOCH = {STEPS_PER_EPOCH}')



    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

    history = model.fit(get_training_dataset(train_dataset), 

                        steps_per_epoch=STEPS_PER_EPOCH, 

                        epochs=EPOCHS, 

                        callbacks=[lr_callback,

                                   save_best_n(fold_num, model),

                                   ],

                        validation_data=get_validation_dataset(val_dataset),

                        class_weight=class_weights,

                        verbose=2,

                       )

    

    idx_sorted=np.argsort(best_score[fold_num])

    best_score[fold_num]=np.array(best_score[fold_num])[idx_sorted]

    best_epoch[fold_num]=np.array(best_epoch[fold_num])[idx_sorted]



    

    

    print(f"\nFold {fold_num} is finished. The best epochs: {[int(best_epoch[fold_num][i]) for i in range(len(best_epoch[fold_num]))]}")

    print(f"The corresponding scores: {[round(best_score[fold_num][i], 5) for i in range(len(best_epoch[fold_num]))]}")



    histories.append(history)
def display_training_curves(fold_num, data):



    plt.figure(figsize=(10,5), facecolor='#F0F0F0')



    epochs=np.arange(1, EPOCHS+1)



    # AUC

    plt.plot(epochs, data['auc'], label='training auc', color='red')

    plt.plot(epochs, data['val_auc'], label='validation auc', color='orange')



    # Loss

    plt.plot(epochs, data['loss'], label='training loss', color='blue')    

    plt.plot(epochs, data['val_loss'], label='validation loss', color='green')



    # Best

    ls=['dotted', 'dashed', 'dashdot', 'solid'] # don't use more than 4 best epochs 

                                                # or make proper adjustments!

    for i in range(NBEST):

        plt.axvline(best_epoch[fold_num][i], 0, 

                    best_score[fold_num][i], linestyle=ls[i], 

                    color='black', label=f'AUC {best_score[fold_num][i]:.5f}')

    

    plt.title(f"Fold {fold_num}. The best epochs: {[int(best_epoch[fold_num][i]) for i in range(len(best_epoch[fold_num]))]}; the best AUC's: {[round(best_score[fold_num][i], 5) for i in range(len(best_epoch[fold_num]))]}.", 

              fontsize='14')

    plt.ylabel('Loss/AUC', fontsize='12')

    plt.xlabel('Epoch', fontsize='12')

    plt.ylim((0, 1))

    plt.legend(loc='lower left')

    plt.tight_layout()

    plt.show()

for fn in range(1, NFOLDS+1):

    display_training_curves(fn, data=histories[fn-1].history)
def setup_test_image(image, data):    

    tab_data=[tf.cast(data[tfeat], dtype=tf.float32) for tfeat in tab_feats]

    tabular=tf.stack(tab_data)



    return {'inp1': image, 'inp2': tabular}

def setup_test_name(image, data):

    return data['image_name']

def get_test_dataset(dataset):

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    

    return dataset

def average_predictions(X, fn):

    

    y_probas=[]

    

    for idx in range(NBEST):

        

        tf.tpu.experimental.initialize_tpu_system(tpu)

        strategy = tf.distribute.experimental.TPUStrategy(tpu)

        gc.collect()



        print(f"Predicting: fold {fn}, model {idx+1} out of {NBEST}...")



        with strategy.scope():

            path_best_model=f'best_model_fold_{fn}_{idx}.hdf5'

            model=tf.keras.models.load_model(SAVE_FOLDER/path_best_model)



        y=model.predict(X)

        y = rankdata(y)/len(y)

        y_probas.append(y)

    

    y_probas=np.average(y_probas, axis=0)



    return y_probas

%%time



preds = pd.DataFrame({'image_name': np.zeros(len(test)), 'target': np.zeros(len(test))})



test_ds = load_dataset(TEST_FNAMES, labeled=False, ordered=True)

test_images_ds = test_ds.map(setup_test_image, num_parallel_calls=AUTO)



test_images_ds = get_test_dataset(test_images_ds)

test_ds = get_test_dataset(test_ds)



test_ids_ds = test_ds.map(setup_test_name, num_parallel_calls=AUTO).unbatch()



preds['image_name'] = next(iter(test_ids_ds.batch(N_TEST_IMGS))).numpy().astype('U')

preds['target'] = np.average([average_predictions(test_images_ds, fn) for fn in range(1, NFOLDS+1)], axis = 0)
sub.head()

del sub['target']

sub = sub.merge(preds, on='image_name')

sub.head()
print(f"The lengths of the submission file and `test` are {len(sub)} and {len(test)}, respectively.")

print(f"The number of NA's in the submission file is {sub.isna().sum().sum()}.")
if colab:

    OUT_FOLDER=SAVE_FOLDER

else:

    OUT_FOLDER=Path('')

    

sub.to_csv(OUT_FOLDER/'submission.csv', index=False)
def setup_val_image(image, label, data, label_name):

    

    tab_data=[tf.cast(data[tfeat], dtype=tf.float32) for tfeat in tab_feats]

    

    tabular=tf.stack(tab_data)



    return {'inp1': image, 'inp2': tabular}

def setup_val_name(image, label, data, label_name):

    return data['image_name']

def setup_val_label(image, label, data, label_name):

    return label

oof= pd.DataFrame({'image_name': train['image_name'].values})



auc=[]

oof_all_folds=[]



for fold_num in range(1, NFOLDS+1):

    

    tf.keras.backend.clear_session()

    

    print("="*50)

    print(f"Starting fold {fold_num}...")    



    print(f"The best epochs: {[int(best_epoch[fold_num][i]) for i in range(len(best_epoch[fold_num]))]}")

    print(f"The corresponding scores: {[round(best_score[fold_num][i], 5) for i in range(len(best_epoch[fold_num]))]}")



    files_val=VAL_FNAMES[f"fold_{fold_num}"]

    

    if debug:

        files_val=files_val[0:2]

    

    val_ds = load_dataset(files_val, ordered = True)

    val_images_ds = val_ds.map(setup_val_image,

                               num_parallel_calls=AUTO)

    val_images_ds = get_validation_dataset(val_images_ds)

        

    val_ds = get_validation_dataset(val_ds)



    val_label_ds = val_ds.map(setup_val_label,

                              num_parallel_calls=AUTO).unbatch()

    val_ids_ds = val_ds.map(setup_val_name,

                            num_parallel_calls=AUTO).unbatch()

    

    n_val_fold = count_data_items(files_val)

    

    print(f'The # of validation files = {n_val_fold}')    

    

    oof_fold= pd.DataFrame()

    

    oof_fold['image_name'] = next(iter(val_ids_ds.batch(n_val_fold))).numpy().astype('U')



    oof_fold['target'] = average_predictions(val_images_ds, fold_num)



    oof_all_folds.append(oof_fold)



    y_true = next(iter(val_label_ds.batch(n_val_fold))).numpy()



    auc_fold=roc_auc_score(y_true, oof_fold['target'].values)

    auc.append(auc_fold)

    

    print(f"Fold {fold_num} is done! ROC AUC = {auc_fold:.5f}")



oof=oof.merge(pd.concat(oof_all_folds), 

              on='image_name', 

              how='left').reset_index(drop=True)



if debug:

    oof=oof.fillna(0)



auc=np.array(auc)

auc_av=auc.mean()

auc_std=auc.std()



print(f"ROC AUC = {auc_av:.5f}, STDEV = {auc_std:.5f} (average across the folds)")



auc_oof=roc_auc_score(train['target'].values, oof['target'].values)



print(f"ROC AUC = {auc_oof:.5f} (out of folds)")
print(f"The lengths of the oof data frame and `train` are {len(oof)} and {len(train)}, respectively.")

print(f"The number of NA's in the oof data frame is {oof.isna().sum().sum()}.")
oof.to_csv(OUT_FOLDER/'oof_1.csv', index=False)