import re, math

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from kaggle_datasets import KaggleDatasets



print("Tensorflow version " + tf.__version__)
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
# Image and batching parameters

IMAGE_SIZE = [512, 512, 3] # at this size, a GPU will run out of memory. Use the TPU

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-%sx%s' % (IMAGE_SIZE[0], IMAGE_SIZE[1]) + '/train/*.tfrec') 

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-%sx%s' % (IMAGE_SIZE[0], IMAGE_SIZE[1]) + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-%sx%s' % (IMAGE_SIZE[0], IMAGE_SIZE[1]) + '/test/*.tfrec')



CLASSES = [

    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 

    'wild geranium', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 

    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 

    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 

    'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 

    'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 

    'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 

    'carnation', 'garden phlox', 'love in the mist', 'cosmos',  'alpine sea holly', 

    'ruby-lipped cattleya', 'cape flower', 'great masterwort',  'siam tulip', 

    'lenten rose', 'barberton daisy', 'daffodil',  'sword lily', 'poinsettia', 

    'bolero deep blue',  'wallflower', 'marigold', 'buttercup', 'daisy', 

    'common dandelion', 'petunia', 'wild pansy', 'primula',  'sunflower', 

    'lilac hibiscus', 'bishop of llandaff', 'gaura',  'geranium', 'orange dahlia', 

    'pink-yellow dahlia', 'cautleya spicata',  'japanese anemone', 'black-eyed susan', 

    'silverbush', 'californian poppy',  'osteospermum', 'spring crocus', 'iris', 

    'windflower',  'tree poppy', 'gazania', 'azalea', 'water lily',  'rose', 

    'thorn apple', 'morning glory', 'passion flower',  'lotus', 'toad lily', 

    'anthurium', 'frangipani',  'clematis', 'hibiscus', 'columbine', 'desert-rose', 

    'tree mallow', 'magnolia', 'cyclamen ', 'watercress',  'canna lily', 

    'hippeastrum ', 'bee balm', 'pink quill',  'foxglove', 'bougainvillea', 

    'camellia', 'mallow',  'mexican petunia',  'bromelia', 'blanket flower', 

    'trumpet creeper',  'blackberry lily', 'common tulip', 'wild rose']



# Training parameters

WARMUP_EPOCHS = 20

WARMUP_LEARNING_RATE = 3e-3

TUNING_EPOCHS = 8

#LR_TUNING = 3e-5

LR_MIN = 1e-10

LR_PEAK = 2e-4

LR_RAMPUP_EPOCHS = 4

LR_DECAY = 0.3



# Random erasing parameters

RE_PROBABILITY = 1

RE_S_LOW = 0.1

RE_S_HIGH = 0.6

RE_RATIO = 0.3



# Random cropping size

CROP_MIN = tf.cast(IMAGE_SIZE[0]*0.6, dtype=tf.int32)

CROP_MAX = tf.cast(IMAGE_SIZE[0]*0.95, dtype=tf.int32)
def random_erasing(image, p=RE_PROBABILITY, sl=RE_S_LOW, sh=RE_S_HIGH, r1=RE_RATIO, r2=1./RE_RATIO):

    

    w = IMAGE_SIZE[0] # image width

    h = IMAGE_SIZE[1] # image height

    c = IMAGE_SIZE[2] # image channels

    s = w*h # image area

    

    p1 = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

    

    erased_image = image

    

    # Chance of applying random erasing

    if p1 <= p:

        while tf.constant(True):

            # Generate random rectangle

            se = s*tf.random.uniform(shape=[], minval=sl, maxval=sh, dtype=tf.float32)

            re = tf.random.uniform(shape=[], minval=r1, maxval=r2, dtype=tf.float32)

            he = tf.cast((se*re)**0.5, tf.int32)

            we = tf.cast((se/re)**0.5, tf.int32)

            xe = tf.random.uniform(shape=[], minval=0, maxval=w, dtype=tf.int32)

            ye = tf.random.uniform(shape=[], minval=0, maxval=h, dtype=tf.int32)

            # If the rectangle fits

            if (xe+we <= w) and (ye+he <= h):

                # Generate blocking rectangle tensor of 0s

                e = tf.zeros(shape=[we, he, c], dtype=tf.int32)

                # Pad blocking rectangle on all 4 sides with 1s to get the full dimension tensor

                e = tf.pad(e, [[xe,w-we-xe], [ye,h-he-ye], [0,0]], mode='CONSTANT', constant_values=1)

                # Multiply padded erasure and image element-wise

                erased_image = tf.math.multiply(tf.cast(image, dtype=tf.float32), tf.cast(e, dtype=tf.float32))

                # Generate rectangle of white noise the same size as the blocking rectangle

                r = tf.random.uniform(shape=[we, he, c], minval=0, maxval=1, dtype=tf.float32) # maxval is excluded of the range

                # Pad the noisy rectangle on all 4 sides with 0s

                r = tf.pad(r, [[xe,w-we-xe], [ye,h-he-ye], [0,0]], mode='CONSTANT', constant_values=0)

                # Add the noisy rectangle and the occluded image

                erased_image = erased_image + r

                break

                

    return tf.cast(erased_image, image.dtype)



def augment_dataset(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = random_erasing(image)

    crop_size = tf.random.uniform(shape=[], minval=CROP_MIN, maxval=CROP_MAX, dtype=tf.int32)

    image = tf.image.random_crop(image, size=[crop_size, crop_size, IMAGE_SIZE[2]])

    image = tf.image.resize(image, size=[IMAGE_SIZE[0], IMAGE_SIZE[1]])

    return image, label
from matplotlib import pyplot as plt



fig, ax = plt.subplots()



blank = tf.constant(1, shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), dtype=tf.float32)

erasure = random_erasing(blank)

ax.imshow(np.asarray(erasure.numpy()))

plt.show()
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, IMAGE_SIZE) # explicit size needed for TPU

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



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(augment_dataset)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    #dataset = dataset.map(augment_dataset)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset



def get_training_dataset_preview(ordered=True):

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    #dataset = dataset.prefetch(AUTO)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)





training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
with strategy.scope():    

    pretrained_model = tf.keras.applications.DenseNet201(

        weights = 'imagenet',

        include_top = False,

        input_shape = IMAGE_SIZE)

    pretrained_model.trainable = False # use transfer learning

    

    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(104, activation = 'softmax')

    ])



warmup_optimizer = tf.keras.optimizers.Adam(lr = WARMUP_LEARNING_RATE)



model.compile(

    optimizer = warmup_optimizer,

    loss = 'sparse_categorical_crossentropy',

    metrics = ['sparse_categorical_accuracy']

)

model.summary()



warmup_history = model.fit(

    training_dataset,

    steps_per_epoch = STEPS_PER_EPOCH,

    epochs = WARMUP_EPOCHS,

    validation_data = validation_dataset)
# Plot accuracy during training

plt.plot(warmup_history.history['sparse_categorical_accuracy'])

plt.plot(warmup_history.history['val_sparse_categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# Plot loss during training

plt.plot(warmup_history.history['loss'])

plt.plot(warmup_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
def scheduler(epoch):

    lr = LR_MIN

    if epoch <= LR_RAMPUP_EPOCHS:

        lr = LR_MIN + epoch*(LR_PEAK-LR_MIN)/LR_RAMPUP_EPOCHS

    else:

        lr = LR_MIN + (LR_PEAK-LR_MIN)*LR_DECAY**(epoch - LR_RAMPUP_EPOCHS)

    return lr
for layer in model.layers:

    layer.trainable = True # Unfreeze layers



tuning_optimizer = tf.keras.optimizers.Adam(lr = LR_PEAK)

#tuning_optimizer = tf.keras.optimizers.Adam(lr = LR_TUNING)



model.compile(

    optimizer = tuning_optimizer,

    loss = 'sparse_categorical_crossentropy',

    metrics = ['sparse_categorical_accuracy']

)

model.summary()



lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)



tuning_history = model.fit(

    training_dataset,

    steps_per_epoch = STEPS_PER_EPOCH,

    epochs = TUNING_EPOCHS,

    validation_data = validation_dataset,

    callbacks = lr_callback)
# Plot accuracy during training

plt.plot(tuning_history.history['sparse_categorical_accuracy'])

plt.plot(tuning_history.history['val_sparse_categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# Plot loss during training

plt.plot(tuning_history.history['loss'])

plt.plot(tuning_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')