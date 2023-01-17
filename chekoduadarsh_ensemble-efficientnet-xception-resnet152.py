



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import os

import seaborn as sns

import cv2

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,  Dropout, BatchNormalization, LeakyReLU,Input

from keras.optimizers import SGD, Adam

from keras.utils import np_utils

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets





# TPU or GPU detection

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

else:

    strategy = tf.distribute.get_strategy()

    

def seed_everything(seed=0):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



seed = 2048

seed_everything(seed)

print("REPLICAS: ", strategy.num_replicas_in_sync)



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()

AUTO = tf.data.experimental.AUTOTUNE

tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
# Configuration

EPOCHS = 40

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMG_SIZE = 700



def format_path(st):

    return GCS_DS_PATH + '/images/' + st + '.jpg'



train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')



train_paths = train.image_id.apply(format_path).values

test_paths = test.image_id.apply(format_path).values

train_labels = train.loc[:, 'healthy':].values

SPLIT_VALIDATION =True

if SPLIT_VALIDATION:

    train_paths, valid_paths, train_labels, valid_labels =train_test_split(train_paths, train_labels, test_size=0.15, random_state=seed)



def decode_image(filename, label=None, IMG_SIZE=(IMG_SIZE, IMG_SIZE)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, IMG_SIZE)

    

    if label is None:

        return image

    else:

        return image, label



def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    if label is None:

        return image

    else:

        return image, label



train_dataset = (

tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .map(data_augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

train_dataset_1 = (

tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .map(data_augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(64)

    .prefetch(AUTO)

)

valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((valid_paths, valid_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_paths)

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

)



    




LR_START = 0.0001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.0001

LR_RAMPUP_EPOCHS = 4

LR_SUSTAIN_EPOCHS = 6

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



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))



import tensorflow as tf



from keras.models import Model

from tensorflow import keras

!pip install -q efficientnet

import efficientnet.tfkeras as efn







with strategy.scope():    

    efficient_net = efn.EfficientNetB7(

                    input_shape=(IMG_SIZE, IMG_SIZE, 3),

                    weights='imagenet',

                    include_top=False

                    )

    x = efficient_net.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model =  keras.Model(inputs = efficient_net.input,outputs=x)

    model.compile(loss="categorical_crossentropy", optimizer= 'adam', metrics=["accuracy"])
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE



history = model.fit(

    train_dataset, 

    epochs=EPOCHS, 

    callbacks=[lr_callback],

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=valid_dataset if SPLIT_VALIDATION else None,

)

predict= model.predict(test_dataset)



prediction = np.ndarray(shape = (test.shape[0],4), dtype = np.float32)

for row in range(test.shape[0]):

    for col in range(4):

        if predict[row][col] == max(predict[row]):

            prediction[row][col] = 1

        else:

            prediction[row][col] = 0

            

prediction = pd.DataFrame(prediction)

prediction.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']

df = pd.concat([test.image_id, prediction], axis = 1)





df.to_csv('effi_submission.csv', index = False)



tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
import tensorflow as tf

from tensorflow.keras.applications import ResNet50V2,ResNet101V2,ResNet152V2,DenseNet201,Xception

from keras.models import Model

from tensorflow import keras

with strategy.scope():    

    Dense_net = Xception(

                    input_shape=(IMG_SIZE, IMG_SIZE, 3),

                    weights='imagenet',

                    include_top=False

                    )

    x = Dense_net.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model =  keras.Model(inputs = Dense_net.input,outputs=x)

    model.compile(loss="categorical_crossentropy", optimizer= 'adam', metrics=["accuracy"])
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE



history = model.fit(

    train_dataset, 

    epochs=EPOCHS, 

    callbacks=[lr_callback],

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=valid_dataset if SPLIT_VALIDATION else None,

)
predict= model.predict(test_dataset)



prediction = np.ndarray(shape = (test.shape[0],4), dtype = np.float32)

for row in range(test.shape[0]):

    for col in range(4):

        if predict[row][col] == max(predict[row]):

            prediction[row][col] = 1

        else:

            prediction[row][col] = 0

            

prediction = pd.DataFrame(prediction)

prediction.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']

df_dense = pd.concat([test.image_id, prediction], axis = 1)





df_dense.to_csv('dense_submission.csv', index = False)



tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
import tensorflow as tf

from tensorflow.keras.applications import ResNet50V2,ResNet101V2,ResNet152V2,DenseNet201

from keras.models import Model

from tensorflow import keras

with strategy.scope():    

    Res_net = ResNet152V2(

                    input_shape=(IMG_SIZE, IMG_SIZE, 3),

                    weights='imagenet',

                    include_top=False

                    )

    x = Res_net.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model =  keras.Model(inputs = Res_net.input,outputs=x)

    model.compile(loss="categorical_crossentropy", optimizer= 'adam', metrics=["accuracy"])
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE



history = model.fit(

    train_dataset, 

    epochs=EPOCHS, 

    callbacks=[lr_callback],

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=valid_dataset if SPLIT_VALIDATION else None,

)
predict= model.predict(test_dataset)



prediction = np.ndarray(shape = (test.shape[0],4), dtype = np.float32)

for row in range(test.shape[0]):

    for col in range(4):

        if predict[row][col] == max(predict[row]):

            prediction[row][col] = 1

        else:

            prediction[row][col] = 0

            

prediction = pd.DataFrame(prediction)

prediction.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']

df_res = pd.concat([test.image_id, prediction], axis = 1)





df_res.to_csv('res_submission.csv', index = False)

def voting(a,b,c):

    if a==b:

        return a

    if b==c:

        return c

    if a==c:

        return a

image_id = df['image_id']

healthy = []

multiple_diseases = []

rust = []

scab = []

for i in range(len(df['healthy'])):

    healthy.append(voting(df['healthy'][i],df_dense['healthy'][i],df_res['healthy'][i]))

    multiple_diseases.append(voting(df['multiple_diseases'][i],df_dense['multiple_diseases'][i],df_res['multiple_diseases'][i]))

    rust.append(voting(df['rust'][i],df_dense['rust'][i],df_res['rust'][i]))

    scab.append(voting(df['scab'][i],df_dense['scab'][i],df_res['scab'][i]))

    

finalsubmission = pd.DataFrame(columns = ['image_id','healthy', 'multiple_diseases', 'rust', 'scab'])



finalsubmission['image_id'] = image_id

finalsubmission['healthy'] = healthy

finalsubmission['multiple_diseases'] = multiple_diseases

finalsubmission['rust'] = rust

finalsubmission['scab'] = scab

finalsubmission.to_csv('submission.csv', index = False)
