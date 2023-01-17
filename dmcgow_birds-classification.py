#from kaggle_datasets import KaggleDatasets

#gcs_path = KaggleDatasets().get_gcs_path('birds-200') 

#gcs_path
!pip install -q efficientnet

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import efficientnet.tfkeras as efn

import glob



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()

keras = tf.keras

layers = keras.layers

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

DATASET_SIZE = 8000

EPOCH = 20

AUTO = tf.data.experimental.AUTOTUNE

print(f'tensorflow version : {tf.__version__}')

print("REPLICAS: ", strategy.num_replicas_in_sync)
image_lib = tf.io.gfile.glob(r'gs://kds-20e52bbc939de4309469253ca7877217ebf6d5ee5bd5b60977b6fa74/birds_train/*/*.jpg')

bird_name = dict((path.split('/')[-2].split('.')[1],int(path.split('/')[-2].split('.')[0])-1) for path in image_lib)

np.save('BirdName.npy', bird_name)

label_dataset = [(int(path.split('/')[-2].split('.')[0])-1) for path in image_lib]

test_lib = tf.io.gfile.glob(r'gs://kds-20e52bbc939de4309469253ca7877217ebf6d5ee5bd5b60977b6fa74/birds_test/*.jpg')

id_dataset = [int(path.split('/')[-1].split('.')[0]) for path in test_lib]
def preprocession(path,label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image,channels=3)

    image = tf.image.resize_with_pad(image,400,400)

    image = tf.image.random_crop(image,[256,256,3])

    image = tf.image.random_flip_left_right(image)

    image = tf.cast(image,tf.float32)

    image = (image - 127.5) / 127.5

    return image,label

def test_preprocession(path,ID):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image,channels=3)

    image = tf.image.resize_with_pad(image,256,256)

    image = tf.cast(image,tf.float32)

    image = (image - 127.5) / 127.5

    return image,ID
dataset = tf.data.Dataset.from_tensor_slices((image_lib,label_dataset))

dataset = dataset.map(preprocession,num_parallel_calls=AUTO).with_options(ignore_order)

dataset = dataset.shuffle(DATASET_SIZE).repeat().batch(BATCH_SIZE).prefetch(AUTO)

test_dataset = tf.data.Dataset.from_tensor_slices((test_lib,id_dataset))

test_dataset = test_dataset.map(test_preprocession,AUTO)

test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTO)
print(dataset)

print(test_dataset)
LR_START = 1e-5

LR_MAX = 1e-3

LR_MIN = 1e-6

LR_RAMPUP_EPOCH = 10

LR_SUSTAIN_EPOCH = 3

LR_EXP_DECAY = 0.8

def lr_schedule(epoch):

    if epoch < LR_RAMPUP_EPOCH:

        lr = LR_START + (LR_MAX - LR_START) / LR_RAMPUP_EPOCH * epoch

    elif epoch < LR_RAMPUP_EPOCH + LR_SUSTAIN_EPOCH:

        lr = LR_MAX

    else:

        lr = LR_MIN + (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCH - LR_SUSTAIN_EPOCH)

    return lr
lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule,verbose=True)

optimizer = keras.optimizers.Adam()

loss = keras.losses.SparseCategoricalCrossentropy()

metrics = keras.metrics.SparseCategoricalAccuracy()
with strategy.scope():

    base_network = efn.EfficientNetB7(input_shape=(256,256,3),weights='imagenet',include_top=False)

    #base_network = keras.applications.InceptionResNetV2(include_top=False,input_shape=[512,512,3])

    #base_network = keras.applications.DenseNet201(include_top=False,input_shape=[512,512,3])

    network = keras.Sequential()

    network.add(base_network)

    network.add(layers.MaxPooling2D())

    network.add(layers.Conv2D(2560,3,padding='same'))

    network.add(layers.BatchNormalization())

    network.add(layers.ReLU())

    network.add(layers.GlobalAveragePooling2D())

    network.add(layers.Dense(1024))

    network.add(layers.BatchNormalization())

    network.add(layers.LeakyReLU())

    network.add(layers.Dense(512))

    network.add(layers.BatchNormalization())

    network.add(layers.LeakyReLU())

    network.add(layers.Dense(200,activation='softmax'))

    network.compile(optimizer=optimizer,loss=loss,metrics=[metrics])

network.summary()
network.fit(dataset,

            epochs=EPOCH,

            steps_per_epoch=DATASET_SIZE//BATCH_SIZE,

            callbacks=[lr_callback])

network.save(r'./Network.h5')
predict_csv = []

for batch_image,batch_id in test_dataset:

    batch_id = batch_id.numpy()

    batch_prediction = network.predict(batch_image)

    batch_prediction = tf.argmax(batch_prediction,1).numpy()

    for i in range(len(batch_id)):

        predict_csv.append([batch_id[i], batch_prediction[i]+1])

    print('*',end='')

csv_name = ['id','label']

csv_data = pd.DataFrame(columns=csv_name,data=predict_csv)

csv_data.to_csv(r'submission.csv',index=False)
print('Done')