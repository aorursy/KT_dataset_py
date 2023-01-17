!pip install efficientnet > /dev/null
import os



import tensorflow as tf

import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets



import numpy as np



tf.__version__
class config:

    GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

    TRAIN_CSV = '../input/siim-isic-melanoma-classification/train.csv'

    TEST_CSV = '../input/siim-isic-melanoma-classification/test.csv'

    TRAIN_FILES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/train*')

    TEST_FILES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/test*')

    VALIDATION_CSV = ""

    

    TOTAL_TRAIN_IMG = 0

    TOTAL_TEST_IMG = 0

    

    IMG_SIZE = [1024, 1024]

    IMG_RESHAPE = [512,512]

    IMG_SHAPE = (512, 512, 3)

    DO_FINETUNE = True

    

    BATCH_SIZE = 8

    BUFFER_SIZE = 100

    EPOCHES = 10 

    

    LOSS = tf.keras.losses.BinaryCrossentropy()

    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01)

    ACCURACY = ['accuracy']

    

    STRATEGY = None

    

    LOG_DIR = './log'

    CHECKPOINT_DIR = './log/checkpoint/cp.cpkt'

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    config.STRATEGY = strategy

    config.BATCH_SIZE = 8 * strategy.num_replicas_in_sync 

else:

    strategy = tf.distribute.get_strategy() 

print("REPLICAS: ", strategy.num_replicas_in_sync)
import pandas as pd

config.TOTAL_TRAIN_IMG = len(pd.read_csv(config.TRAIN_CSV).image_name)

config.TOTAL_TEST_IMG = len(pd.read_csv(config.TEST_CSV).image_name)
def get_model():

    model = tf.keras.Sequential([

                efn.EfficientNetB0(

                    input_shape=config.IMG_SHAPE,

                    weights='imagenet',

                    include_top=False

                ),

                tf.keras.layers.GlobalAveragePooling2D(),

                tf.keras.layers.Dense(1, activation='sigmoid')

            ])



    return model
## Helper Functions

def process_training_data(data_file):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), 

        "target": tf.io.FixedLenFeature([], tf.int64),

    }

    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)

    img = tf.image.decode_jpeg(data['image'], channels=3)

    img = tf.cast(img, tf.float32) / 255.0

    img = tf.reshape(img, [*config.IMG_SIZE, 3])

    img = tf.image.resize(img, config.IMG_RESHAPE)



    

    label = tf.cast(data['target'], tf.int32)



    return img, label



def process_test_data(data_file):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string),

    }

    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)

    img = tf.image.decode_jpeg(data['image'], channels=3)

    img = tf.cast(img, tf.float32) / 255.0

    img = tf.reshape(img, [*config.IMG_SIZE, 3])

    img = tf.image.resize(img, config.IMG_RESHAPE)



    

    idnum = data['image_name']



    return img, idnum
def fit_engine(model,dataset):

    model.compile(

            optimizer=config.OPTIMIZER, 

            loss=config.LOSS, 

            metrics=config.ACCURACY

        )

    history = model.fit(

        dataset, 

        epochs=config.EPOCHES, 

        steps_per_epoch=(config.TOTAL_TRAIN_IMG//config.BATCH_SIZE),

    )



    return history
def run():

    #Creating Dataset

    dataset = (

        tf.data.TFRecordDataset(

            config.TRAIN_FILES,  

            num_parallel_reads=tf.data.experimental.AUTOTUNE

        ).map(

            process_training_data,

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        ).repeat(

        ).shuffle(

            buffer_size=config.BUFFER_SIZE

        ).batch(

            config.BATCH_SIZE

        ).prefetch(

            tf.data.experimental.AUTOTUNE

        )

    )

    

    #Setup model and train

    if config.STRATEGY is not None:

        with strategy.scope():

            model = get_model()

    else:

        model = get_model()

        

    history = fit_engine(model, dataset)

        

    return model, history
# model, history = run()
test_dataset = (

    tf.data.TFRecordDataset(

        config.TEST_FILES,  

        num_parallel_reads=tf.data.experimental.AUTOTUNE

    ).map(

        process_test_data,

        num_parallel_calls=tf.data.experimental.AUTOTUNE

    ).batch(

        config.BATCH_SIZE

    ).prefetch(

        tf.data.experimental.AUTOTUNE

    )

)



test_imgs = test_dataset.map(lambda images, ids: images)

img_ids_ds = test_dataset.map(lambda images, ids: ids).unbatch()



predictions = model.predict(test_imgs).flatten()



img_ids = []

for coutner, ids in enumerate(img_ids_ds):

    if coutner%500 == 0:

        print(coutner)

    img_ids.append(ids.numpy())



img_ids = np.array(img_ids).astype('U')



np.savetxt(

    'sample_submission.csv', 

    np.rec.fromarrays([img_ids, predictions]), 

    fmt=['%s', '%f'], 

    delimiter=',', 

    header='image_name,target', 

    comments=''

)
pd.read_csv('sample_submission.csv').head()