# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install tfa-nightly
!pip install -q pyyaml h5py
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
from matplotlib import pyplot as plt
import numpy as np
import random 
import tensorflow_addons as tfa
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on Tpu" , tpu.master())
except ValueError as e:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:" , strategy.num_replicas_in_sync)
        
random.seed(1)
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False
LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "class": tf.io.FixedLenFeature([], tf.int64)}

UNLABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string), 
    "id": tf.io.FixedLenFeature([], tf.string), 
    
}
IMAGE_SIZE = [512,512]
EPOCHS = 20
BATCH_SIZE = 16 * strategy.num_replicas_in_sync 
training_data = tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords-jpeg-512x512/train/*.tfrec")
validation_data = tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords-jpeg-512x512/val/*.tfrec")
testing_data = tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords-jpeg-512x512/test/*.tfrec")
NUM_CLASSES = 104
NUM_TRAINING_IMAGES = 12753
NUM_TEST_IMAGES = 7382
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
def flip(x: tf.Tensor) -> tf.Tensor:
    check = random.randint(0,9)
    
    if check < 5:
        x = tf.image.random_flip_left_right(x)
        return x
    
    x = tf.image.random_flip_up_down(x)

    return x



def color(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x 



def rotate(x: tf.Tensor) -> tf.Tensor:
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))



def shear(x: tf.Tensor) -> tf.Tensor:
    check = random.randint(0 , 5)
    
    if check > 3:
        x = tfa.image.shear_x(x , 0.2 , 0)
        return x
    
    x = tfa.image.shear_y(x , 0.2 , 0)
    
    return x

def random_all(x:tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
def augment_data(image , label):
    
    check = random.randint(0,9)
    
    if check == 1:
        image = flip(image)
        
    elif check == 2:
        image = color(image)
    
    elif check == 3:
        image = rotate(image)
    
    elif check == 4:
        image = shear(image)
    
    elif check == 5:
        image = random_all(image)
    
    return image , label
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image
def read_labeled_tfrecord(record):
    record = tf.io.parse_single_example(record , LABELED_TFREC_FORMAT)
    image = decode_image(record['image'])
    label = tf.cast(record['class'] , tf.int32)
    return image , label
    
def read_unlabeled_tfrecord(record):
    record = tf.io.parse_single_example(record , UNLABELED_TFREC_FORMAT)
    image = decode_image(record['image'])
    id_num = record['id'] 
    return image , id_num
def load_dataset(filenames , labeled=True , ordered = False):
    if not ordered:
        ignore_order.experimental_deterministic = False
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)
    return dataset
def load_augmented_dataset(filenames , labeled=True , ordered = False):
    if not ordered:
        ignore_order.experimental_deterministic = False
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)
    dataset = dataset.map(augment_data)
    return dataset
def get_training_dataset():
    dataset = load_dataset(training_data , labeled = True , ordered = False)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset
def get_augmented_dataset():
    dataset = load_augmented_dataset(training_data , labeled = True , ordered = False)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset
def get_validation_dataset():
    dataset = load_dataset(validation_data , labeled = True , ordered = False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    return dataset 
def get_test_dataset(ordered = False):
    dataset = load_dataset(testing_data , labeled = False , ordered = ordered)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset 
training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()
!pip install efficientnet
import efficientnet.tfkeras as efn

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss' , patience = 4)

learning_rate_start =  0.00001

learning_rate_max = 0.00005 * strategy.num_replicas_in_sync

learning_rate_min = 0.0001

learning_rate_boost_epochs = 3

learning_rate_sustain_epochs = 0 

learning_rate_decay = 0.9

def learning_rate_schedule(epoch):
    if epoch < learning_rate_boost_epochs:
        
        lr = (learning_rate_max - learning_rate_start) / learning_rate_boost_epochs * epoch + learning_rate_start
        
    elif epoch < learning_rate_boost_epochs + learning_rate_sustain_epochs:
        
        lr = learning_rate_max
        
    else:
        
        lr = (learning_rate_max - learning_rate_min) * learning_rate_decay **(epoch - learning_rate_boost_epochs - learning_rate_sustain_epochs) + learning_rate_min
        
    return lr


learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule , verbose = True)

def create_model():
    with strategy.scope():
        input_layer = tf.keras.layers.Input(shape = (*IMAGE_SIZE,3))

        pretrained_model = efn.EfficientNetB7(include_top = False , weights = 'noisy-student' , input_shape = (*IMAGE_SIZE,3) , input_tensor = input_layer , pooling='avg')   

        for layer in pretrained_model.layers:
            layer.trainable = True

        X = tf.keras.layers.Dropout(0.2)(pretrained_model.layers[-1].output)

        X = tf.keras.layers.Dense(NUM_CLASSES , activation = 'softmax' , dtype= 'float32')(X)


        model = tf.keras.Model(inputs = input_layer , outputs = X)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False)

        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        model.compile(loss = loss , optimizer = optimizer , metrics=['sparse_categorical_accuracy'])
        
        return model
model = create_model()
model.summary()
model.fit(training_dataset , epochs = EPOCHS , validation_data = validation_dataset , steps_per_epoch = STEPS_PER_EPOCH , callbacks = [learning_rate_callback] )

augmented_dataset = get_augmented_dataset()
model.fit(augmented_dataset , epochs = EPOCHS , validation_data = validation_dataset , steps_per_epoch = STEPS_PER_EPOCH , callbacks = [learning_rate_callback] )
test_dataset = get_test_dataset(ordered=True)
print("Predicting")
test_images_ds = test_dataset.map(lambda image , idnum : image)
prob = model.predict(test_images_ds)
pred = np.argmax(prob , axis = -1)
print(pred)
print("Generating Csv")
test_ids_ds = test_dataset.map(lambda image , idnum : idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
np.savetxt('submission.csv' , np.rec.fromarrays([test_ids , pred]), fmt=['%s' , '%d'] , delimiter=',',header='id,label' , comments='')
!head submission.csv
