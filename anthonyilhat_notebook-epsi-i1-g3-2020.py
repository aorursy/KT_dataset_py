import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from kaggle_datasets import KaggleDatasets
import os
import re
from sklearn.model_selection import train_test_split
SEED = 123

np.random.seed(SEED)
tf.random.set_seed(SEED)

DEVICE = "TPU"
BASEPATH = "../input/siim-isic-melanoma-classification"
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE
df_train = pd.read_csv(os.path.join(BASEPATH, 'train.csv'))
df_test = pd.read_csv(os.path.join(BASEPATH, 'test.csv'))

GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
TRAINING_FILENAMES = np.array(tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec'))
TEST_FILENAMES = np.array(tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec'))

CLASSES = [0,1]   
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, [*IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    #label = tf.cast(example['class'], tf.int32)
    label = tf.cast(example['target'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['image_name']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


##### FONCTION A DEVELOPPER POUR AMELIORER LE MODELE

def data_augment(image, label):
    
    image = tf.image.convert_image_dtype(image,tf.float32)
    image  =tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image,max_delta=0.5)  

    return image, label   

def get_training_dataset(augment = True):
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    
    if augment == True:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(SEED)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def display_training_curves(training, validation, title, subplot):
    """
    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
    """
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(20,15), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
    
def prediction_test_csv(model,nom_model, df_sub):
    
    test_ds = get_test_dataset(ordered=True)
    print('Computing predictions...')
    test_images_ds = test_ds.map(lambda image, idnum: image)
    probabilities = model.predict(test_images_ds)
    print('Generating submission.csv file...')
    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
    pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
    pred_df.head()
    del df_sub['target']
    df_sub = df_sub.merge(pred_df, on='image_name')
    #sub.to_csv('submission_label_smoothing.csv', index=False)
    df_sub.to_csv('submission_' + nom_model + '.csv', index=False)
    print(df_sub.head())
EPOCHS = 50 # le nombre d'itération pour l'apprentissage du modèle
BATCH_SIZE = 8 * strategy.num_replicas_in_sync # le nombre d'images traitées à la fois
IMAGE_SIZE = [32,32] # liste [hauteur, largeur] de l'image
IMAGE_CHANNEL = 3 # 1 en gris, 3 en couleur
LR = 0.0001 # le taux d'apprentissage
TRAINING_FILENAMES,VALIDATION_FILENAMES = train_test_split(TRAINING_FILENAMES,test_size = 0.2,random_state = SEED)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
with strategy.scope():
    
    lenet5_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, (5,5), activation='relu', input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1], IMAGE_CHANNEL)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, (5,5), activation='relu'),# une autre couche de convolution: 16 filtres 5x5, également avec une activation relu. Ne pas spécifier de format d'entrée (input shape)
        tf.keras.layers.MaxPooling2D(),# une autre couche maxpooling 2D
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),# une couche de neurones tf.keras.layers.Dense: 120 neurones, activation relu
        tf.keras.layers.Dense(120, activation='relu'),# une couche de neurones tf.keras.layers.Dense: 120 neurones, activation relu
        tf.keras.layers.Dense(84, activation='relu'),# une couche de neurones tf.keras.layers.Dense: 84 neurones, activation relu
        #tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    lenet5_model.summary()
    

    adam = tf.keras.optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    lenet5_model.compile(loss=loss, metrics=[tf.keras.metrics.AUC(name='auc')],optimizer=adam)
history = lenet5_model.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS, validation_data=get_validation_dataset())
display_training_curves(
    history.history['loss'], 
    history.history['val_loss'], 
    'loss', 311)
display_training_curves(
    history.history['auc'], 
    history.history['val_auc'], 
    'auc', 312)
df_sub = pd.read_csv(os.path.join(BASEPATH, 'sample_submission.csv'))
nom_model = 'lenet5' #servira simplement à nommer votre fichier excel
prediction_test_csv(lenet5_model,nom_model, df_sub)