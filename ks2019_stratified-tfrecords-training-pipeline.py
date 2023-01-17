import numpy as np, pandas as pd, os

import matplotlib.pyplot as plt, cv2

import tensorflow as tf, re, math

from tqdm import tqdm
RESIZE = True

IMAGE_SIZE = 224

N_GROUPS = 12

N_FOLDS = 5

N_TFRs = N_GROUPS*N_FOLDS

SUBSET = False  # Keep SUBSET=True while debugging (Faster Execution)

SUBSET_SIZE = 10000

BATCH_SIZE = 32

FOLDS = [0]

GROUPS = [11]

assert max(FOLDS)<N_FOLDS, "ELEMENTS OF FOLDS can't be greater than N_FOLDS"

assert max(GROUPS)<N_GROUPS, "ELEMENTS OF FOLDS can't be greater than N_FOLDS"
train_df = pd.read_csv('../input/landmark-recognition-2020/train.csv')

train_df['original_landmark_id'] = train_df.landmark_id

print(train_df.shape)

train_df['order'] = np.arange(train_df.shape[0])

train_df['order'] = train_df.groupby('landmark_id').order.rank()-1

landmark_counts = train_df.landmark_id.value_counts()

train_df['landmark_counts'] = landmark_counts.loc[train_df.landmark_id.values].values

train_df['fold'] = (train_df['order']%N_FOLDS).astype(int)

all_groups = [(1/N_GROUPS)*x for x in range(N_GROUPS)]



print(train_df.landmark_counts.quantile(all_groups))

for i,partition_val in enumerate(train_df.landmark_counts.quantile(all_groups).values):

                     train_df.loc[train_df.landmark_counts>=partition_val,'group'] = i 

        

landmark_map = train_df.sort_values(by='landmark_counts').landmark_id.drop_duplicates().reset_index(drop=True)

landmark_dict = {landmark_map.loc[x]:81312-x for x in range(81313)}

train_df['landmark_id'] = train_df.original_landmark_id.apply(lambda x: landmark_dict[x])

train_df = train_df.sample(frac=1).reset_index(drop=True)

train_df.to_csv('train_meta_data.csv',index=False)

train_df.sample(10)
#Checking Null values

train_df.isna().sum().sum()
train_df.groupby('group').landmark_counts.agg(['min','max'])
#Landmark Counts

train_df.landmark_id.value_counts()
#No of images GroupBy landmark counts

train_df.landmark_counts.value_counts()
#No of Images in Each Folds

train_df.fold.value_counts()
#No of Images in Each Group

train_df.group.value_counts()
# No of Landmark in each Fold

train_df.drop_duplicates(['fold','landmark_id']).groupby('fold').landmark_id.count()
train_df.groupby(['fold','group']).id.count()
def _bytes_feature(value):

  """Returns a bytes_list from a string / byte."""

  if isinstance(value, type(tf.constant(0))):

    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _float_feature(value):

  """Returns a float_list from a float / double."""

  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def _int64_feature(value):

  """Returns an int64_list from a bool / enum / int / uint."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def serialize_example(image,image_id,landmark_id):

    feature = {

        'image': _bytes_feature(image),

        'image_id': _bytes_feature(image_id),

        'landmark_id': _int64_feature(landmark_id),

      }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
if SUBSET:

    train_df = train_df.sample(SUBSET_SIZE)

for fold in FOLDS:

    for group in GROUPS:

        tfr_filename = 'train-{}-{}.tfrec'.format(fold,group)

        print("Writing",tfr_filename)

        with tf.io.TFRecordWriter(tfr_filename) as writer:

            indices = train_df[(train_df.fold==fold) & (train_df.group==group)].index.to_list()

            for index in tqdm(indices):

                image_id = train_df.loc[index,'id']

                landmark_id = train_df.loc[index,'landmark_id']

                image_path = "../input/landmark-recognition-2020/train/{}/{}/{}/{}.jpg".format(image_id[0],image_id[1],image_id[2],image_id) 

                image = cv2.imread(image_path)

                if RESIZE:

                    image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))

                image = cv2.imencode('.jpg', image, (cv2.IMWRITE_JPEG_QUALITY, 100))[1].tostring()

                image_id = str.encode(image_id)

                sample = serialize_example(image,image_id,landmark_id)

                writer.write(sample)
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE_, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        'landmark_id': tf.io.FixedLenFeature([], tf.int64),

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = example['landmark_id']

    return image, label # returns a dataset of (image, label) pairs



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
IMAGE_SIZE_ = [IMAGE_SIZE,IMAGE_SIZE]

AUTO = tf.data.experimental.AUTOTUNE

TRAINING_FILENAMES = tf.io.gfile.glob('train*.tfrec')

print(TRAINING_FILENAMES)

dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

dataset = dataset.repeat()

dataset = dataset.shuffle(2048)

dataset = dataset.batch(BATCH_SIZE)

dataset = dataset.prefetch(AUTO) #This dataset can directly be passed to keras.fit method
# numpy and matplotlib defaults

np.set_printoptions(threshold=15, linewidth=80)

CLASSES = [0,1]



def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    #if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

    #    numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



def display_single_sample(image, label, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    title = str(label)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)

    

def display_batch_of_images(databatch):

    """

    Display single batch Of images 

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle

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

        correct = True

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_single_sample(image, label, subplot, not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()
# Displaying single batch of TFRecord

train_batch = iter(dataset)

display_batch_of_images(next(train_batch))
!pip install -q efficientnet >> /dev/null

!pip -qq install focal-loss >> /dev/null

import pandas as pd, numpy as np, gc,os

from kaggle_datasets import KaggleDatasets

import tensorflow as tf, re, math

import tensorflow.keras.backend as K

import efficientnet.tfkeras as efn

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt, cv2

from focal_loss import BinaryFocalLoss

from tqdm import tqdm

os.listdir('../input')
DEVICE  = 'TPU'

meta_path = '../input/landmark-tfr1-metadata/train_meta_data.csv'

IMAGE_SIZE = 224

N_GROUPS = 12

N_FOLDS = 5

N_TFRs = N_GROUPS*N_FOLDS

BATCH_SIZE = 32

EPOCHS = 4

EFF_NET = 0

initial_weights = 1

lossfn = 0

gamma = 2

GROUPS = [9,10,11]

VAL_FOLDS = [0]

TRAIN_FOLDS = [1,2,3,4]

SMOOTHING_FACTOR = 0

assert max(GROUPS)<N_GROUPS, "ELEMENTS OF GROUPS can't be greater than N_GROUPS"
if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    



AUTO     = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')
train_df = pd.read_csv(meta_path)

train_df = train_df[(train_df.group.isin(GROUPS))&(train_df.fold.isin(TRAIN_FOLDS+VAL_FOLDS))].reset_index(drop=True)

train_size = train_df.fold.isin(TRAIN_FOLDS).sum()

val_size = train_df.fold.isin(VAL_FOLDS).sum()

N_CATEGORIES = train_df.landmark_id.max()

print(train_df.shape,N_CATEGORIES,train_size,val_size)

train_df.sample(5)
def decode_image(image_data):



    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE_, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        'landmark_id': tf.io.FixedLenFeature([], tf.int64),

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.one_hot(tf.cast(example['landmark_id'], tf.int32),N_CATEGORIES)

    return image, label # returns a dataset of (image, label) pairs



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls = AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset
all_gcs_paths = ['gs://kds-fd5727aa3747d7eb73196142c44fa93a71a5b7521c0561d89f457ac6',

 'gs://kds-189bb71a7d382041c75fcea19b6e2abd7bc22f5dd572ed0d2407341a',

 'gs://kds-255282f43cc7e3685b4486bff58e5a6fb4aa38e4eb1851e116090c36',

 'gs://kds-7178be73016f5b6ce11e99458d46bff145ab4d2d37356a88afb50b12',

 'gs://kds-904cfff103eb2a2b529a94fd37dbe6c42ab2338b9960d668a601b593',

 'gs://kds-2ccd1eb20077ad466f123d93233a3be476f48df40be35f75f2b182d1',

 'gs://kds-59f19a86ca314f1e2f18343cf3207bdb2c17390a46535b189a6b4328',

 'gs://kds-ac34da5d55984e9ec1417386a2df8259c4afa004d323b3c8095739f5',

 'gs://kds-284baa78992914eaafc5a2399aa5d7bddb11a8c801bc5aae0e697bb6',

 'gs://kds-ec05b4e3b870a96dfbc42994dd73464c68b8e86cb331870993ec82b8',

 'gs://kds-2eefb372303179252f56b66e38e07269fba8c3ae75e311d56268baeb',

 'gs://kds-f16dd5ab80b1cda4fe48c6088809e2bbeda8f2e9b28bd868dc55ea8b',

 'gs://kds-0632a5466c5ed260694bb9689e0a472b46a55c44c2ba005a23a482ea',

 'gs://kds-6be36c39f54f9b78a683c836df9f889c92e4f53c518528f774b9931e',

 'gs://kds-df6221ddab9b7479d3808fd411d6506d2f3c7de480ad3b9bc787695c',

 'gs://kds-859b6ea2c2d48b1d6d22dfacd5272f2ce45e4810fc7779fd6f7f4713',

 'gs://kds-224a05475e78b62a0777154a188de1c0a7fef9030635ba0125dba916',

 'gs://kds-870f5955cec5fdecc43571e30fe6270c2f9a15a2e7b2ae519e9e3860',

 'gs://kds-2cfdad8a024924213ac55c6109b893192fdd6b1bcb59f67f69b7d6ee',

 'gs://kds-00fcf91887605579f838b198605ae39a8d8ac4a0d172343a1614a367']
IMAGE_SIZE_ = [IMAGE_SIZE,IMAGE_SIZE]

AUTO = tf.data.experimental.AUTOTUNE

ALL_FILENAMES = []

for folder in all_gcs_paths:

    ALL_FILENAMES += tf.io.gfile.glob('{}/train*.tfrec'.format(folder))
TRAIN_FILENAMES = list(filter(lambda x:(int(x.split('/')[-1][6]) in TRAIN_FOLDS) and

                              (int(x.split('/')[-1].replace('.','-').split('-')[-2]) in GROUPS),ALL_FILENAMES))

VALIDATION_FILENAMES = list(filter(lambda x:(int(x.split('/')[-1][6]) in VAL_FOLDS) and

                              (int(x.split('/')[-1].replace('.','-').split('-')[-2]) in GROUPS),ALL_FILENAMES))

len(TRAIN_FILENAMES),len(VALIDATION_FILENAMES)
def get_dataset(FILENAMES):

    dataset = load_dataset(FILENAMES, labeled=True)

    dataset = dataset.cache()

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE*REPLICAS)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



train_dataset = get_dataset(TRAIN_FILENAMES)

validation_dataset = get_dataset(VALIDATION_FILENAMES)
class ArcMarginProduct(tf.keras.layers.Layer):

    '''

    Implements large margin arc distance.



    Reference:

        https://arxiv.org/pdf/1801.07698.pdf

        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/

            blob/master/src/modeling/metric_learning.py

    '''

    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,

                 ls_eps=0.0, **kwargs):



        super(ArcMarginProduct, self).__init__(**kwargs)



        self.n_classes = n_classes

        self.s = s

        self.m = m

        self.ls_eps = ls_eps

        self.easy_margin = easy_margin

        self.cos_m = tf.math.cos(m)

        self.sin_m = tf.math.sin(m)

        self.th = tf.math.cos(math.pi - m)

        self.mm = tf.math.sin(math.pi - m) * m



    def build(self, input_shape):

        super(ArcMarginProduct, self).build(input_shape[0])



        self.W = self.add_weight(

            name='W',

            shape=(int(input_shape[0][-1]), self.n_classes),

            initializer='glorot_uniform',

            dtype='float32',

            trainable=True,

            regularizer=None)



    def call(self, inputs):

        X, y = inputs

        y = tf.cast(y, dtype=tf.int32)

        cosine = tf.matmul(

            tf.math.l2_normalize(X, axis=1),

            tf.math.l2_normalize(self.W, axis=0)

        )

        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:

            phi = tf.where(cosine > 0, phi, cosine)

        else:

            phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = tf.cast(

            tf.one_hot(y, depth=self.n_classes),

            dtype=cosine.dtype

        )

        if self.ls_eps > 0:

            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes



        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.s

        return output

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 

        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6,efn.EfficientNetB7]





def build_model(dim=128, ef=0):

    if initial_weights == 0: weights = 'noisy-student'

    else: weights = 'imagenet'

    inp = tf.keras.layers.Input(shape=(dim,dim,3))

    base = EFNS[ef](input_shape=(dim,dim,3),weights= weights,include_top=False)

    x = base(inp)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(N_CATEGORIES,activation='softmax')(x)

#     margin = ArcMarginProduct(

#         n_classes=n_classes,

#         s=scale,

#         m=margin,

#         name='head/arc_margin',

#         dtype='float32')

    

    model = tf.keras.Model(inputs=inp,outputs=x)

#     opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4 / EPOCHS)

    model.compile(

      optimizer = opt,

      loss = [tf.keras.losses.CategoricalCrossentropy()],

      metrics = [tf.keras.metrics.CategoricalAccuracy()]

    ) 

    return model
LR_START = 0.00001

LR_MAX = 0.0001 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 15

LR_SUSTAIN_EPOCHS = 3

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
# BUILD MODEL

K.clear_session()

with strategy.scope():

    model = build_model(dim=IMAGE_SIZE,ef=EFF_NET)

model.summary()
sv = tf.keras.callbacks.ModelCheckpoint(

        'best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True,

        save_weights_only=True, mode='min', save_freq='epoch')

# lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',  mode = 'min', factor = 0.5, patience = 2, verbose = 1, min_delta = 0.0001)



history = model.fit(train_dataset,epochs=EPOCHS,callbacks = [sv,lr_callback],validation_data=validation_dataset,

                    steps_per_epoch=train_size/BATCH_SIZE//REPLICAS, validation_steps=val_size/BATCH_SIZE//REPLICAS, 

                    verbose=True)

                    