from IPython.display import clear_output
!pip install -U efficientnet

clear_output()
import math, re, os, gc, random

import tensorflow as tf

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
import efficientnet.tfkeras as efn

from matplotlib import pyplot as plt
#TPU or GPU detection

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

!gsutil ls $GCS_DS_PATH/images/

clear_output()
img = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_0.jpg')

print(img.shape)
train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

sub_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
train_df.head()
test_df.head()
img_size = 1024 #(Trying out 512+256+128)

IMAGE_SIZE = [1024,1024]

#img_size = 1000

EPOCHS = 20

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

n_classes = 4

CLASSES = 4
#inspired from https://www.kaggle.com/ateplyuk/fork-of-plant-2020-tpu-915e9c



LR_START = 0.00001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

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
train_tpu_paths = train_df.image_id.apply(lambda x: GCS_DS_PATH+"/images/"+x+".jpg").values

train_labels = train_df.iloc[:, 1:].values

test_tpu_paths = test_df.image_id.apply(lambda x: GCS_DS_PATH+"/images/"+x+".jpg").values
def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    if label is None:

        return image

    else:

        return image, label

    

def data_augment(image, label=None, seed=2020):

    

    image = tf.image.random_flip_left_right(image, seed=seed)

    image = tf.image.random_flip_up_down(image, seed=seed)

    

    if tf.random.uniform([1])>0.4:

        image = tf.image.adjust_brightness(image, 0.2)

        

    

    if label is None:

        return image

    else:

        return image, label
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

        

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))
def transform(image,label):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    rot = 15. * tf.random.normal([1],dtype='float32')

    shr = 5. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    h_shift = 16. * tf.random.normal([1],dtype='float32') 

    w_shift = 16. * tf.random.normal([1],dtype='float32') 

  

    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3]),label
train_set = (tf.data.Dataset

            .from_tensor_slices((train_tpu_paths, train_labels))

            .map(decode_image, num_parallel_calls=AUTO)

            .map(data_augment, num_parallel_calls=AUTO)

            .map(transform, num_parallel_calls=AUTO)

            .repeat()

            .shuffle(512)

            .batch(BATCH_SIZE)

            .prefetch(AUTO))



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_tpu_paths)

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

)
with strategy.scope():

    enet = efn.EfficientNetB7(

        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

        weights='noisy-student',

        include_top=False

    )



    enet.trainable = True



    model = tf.keras.Sequential([

        enet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(CLASSES, activation='softmax')

    ])



model.compile(

    optimizer='adam',

    loss = 'categorical_crossentropy',

    metrics=['categorical_accuracy']

)
%%time



hist = model.fit(train_set,

         steps_per_epoch=train_labels.shape[0]//BATCH_SIZE,

         epochs = EPOCHS,

         callbacks=[lr_callback],

         )
%%time



preds = model.predict(test_dataset)

preds
with strategy.scope():

    gc.collect()
preds.shape
sub_df.shape
sub_df.iloc[:, 1:]=preds
sub_df.to_csv("submission.csv", index=False)
su=pd.read_csv("submission.csv")

su