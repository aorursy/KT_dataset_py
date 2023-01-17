import numpy as np

import pandas as pd

import cv2

import tensorflow as tf

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets

%matplotlib inline

scale_percent = 30

# width = int(1365 * scale_percent / 100)

# height = int(2048 * scale_percent / 100)

width = 512

height = 512

dim = (width, height)

dim
AUTO = tf.data.experimental.AUTOTUNE



tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



batch_size = 16*tpu_strategy.num_replicas_in_sync
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

GCS_DS_PATH
def ImageIDtodir(label):

    return GCS_DS_PATH+'/images/' + label + '.jpg'

train_df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

train_dir = train_df['image_id'].apply(ImageIDtodir).values

test_df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

test_dir = test_df['image_id'].apply(ImageIDtodir).values

y =  train_df.loc[:, 'healthy':].values

train_dir, valid_dir, y_train, y_val = train_test_split(train_dir,y,test_size=0.15)
def load_image(filename, label=None,image_size=(width,height)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image,image_size)   

    if label is None:

        return image

    else:

        return image, label

    



def augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset.from_tensor_slices((train_dir, y_train))

    .map(load_image, num_parallel_calls=AUTO)

    .cache()

    .map(augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(batch_size)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset.from_tensor_slices((valid_dir,y_val))

    .map(load_image, num_parallel_calls=AUTO)

    .cache()

    .batch(batch_size)

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_dir)

    .map(load_image, num_parallel_calls=AUTO)

    .cache()

    .batch(batch_size)

)
with tpu_strategy.scope():

    DenseNetModel = tf.keras.applications.DenseNet121(input_shape=(width,height,3),

                                              weights='imagenet',

                                              include_top=False)

    model = tf.keras.models.Sequential(

        [DenseNetModel,

#             tf.keras.layers.Convolution2D(1024,(5,5),strides=(2,2),activation='relu'),

#             tf.keras.layers.Convolution2D(2048,(4,4),strides=(2,2),activation='relu'),

#             tf.keras.layers.AveragePooling2D((5,5),2),

#             tf.keras.layers.AveragePooling2D((4,4),2),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(50, activation='relu'),

        tf.keras.layers.Dense(4, activation='softmax')]

    )

    # Compile the model

    model.compile(loss='categorical_crossentropy',

                optimizer='adam',

                metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
model.summary()
# Learning rate schedular

def lr_scheduler(epoch, lr):

    decay_rate = 0.5

    decay_step = 5

    if epoch % decay_step == 0 and epoch:

        return lr * decay_rate

    return lr

callbacks = [

    tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

]
history = model.fit(train_dataset,

                    epochs=25,

                    callbacks=callbacks,

                    steps_per_epoch=y.shape[0] // batch_size,

                    validation_data=valid_dataset)
plt.plot(history.history['loss'])

plt.title('Model accuracy')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()
predictions = model.predict(test_dataset,verbose=1)
temp_df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

submission = pd.DataFrame(

    {'image_id': temp_df['image_id'],

     'healthy': predictions[:,0],

     'multiple_diseases': predictions[:,1],

     'rust': predictions[:,2],

     'scab': predictions[:,3],

    })

submission.to_csv("plant_pathology.csv",index=False)

submission