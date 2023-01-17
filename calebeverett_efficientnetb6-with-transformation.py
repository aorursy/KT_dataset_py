%matplotlib inline

%load_ext autoreload

%autoreload 2
from google.cloud import storage

import json

import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

from pathlib import Path

from sklearn.metrics import classification_report

import subprocess

import sys

import tensorflow as tf

import time

from tqdm.notebook import tqdm



from tensorflow.keras.backend import dot
def print_output(output):

    """Prints output from string."""

    for l in output.split('\n'):

        print(l)



def print_pred_metrics(label_actual, label_pred):

    """Prints prediction evaluation metrics and report."""

    print(classification_report(label_actual, label_pred))

#     print(pd.crosstab(label_actual, label_pred, margins=True))

        

def run_command(command):

    """Runs command line command as a subprocess returning output as string."""

    STDOUT = subprocess.PIPE

    process = subprocess.run(command, shell=True, check=False,

                             stdout=STDOUT, stderr=STDOUT, universal_newlines=True)

    return process.stdout



def show_images(imgs, titles=None, hw=(3,3), rc=(4,4)):

    """Show list of images with optiona list of titles."""

    h, w = hw

    r, c = rc

    fig=plt.figure(figsize=(w*c, h*r))

    gs1 = gridspec.GridSpec(r, c, fig, hspace=0.2, wspace=0.05)

    for i in range(r*c):

        img = imgs[i].squeeze()

        ax = fig.add_subplot(gs1[i])

        if titles != None:

            ax.set_title(titles[i], {'fontsize': 10})

        plt.imshow(img)

        plt.axis('off')

    plt.show()
output = run_command('pip freeze | grep efficientnet')

if output == '':

    print_output(run_command('pip install efficientnet'))

else:

    print_output(output)

from efficientnet import tfkeras as efn
KAGGLE = os.getenv('KAGGLE_KERNEL_RUN_TYPE') != None



BUCKET = 'flowers-caleb'

client = storage.Client(project='fastai-caleb')

bucket = client.get_bucket(BUCKET)



if KAGGLE:

    from kaggle_datasets import KaggleDatasets

    DATASET_DIR = Path('/kaggle/input/flowers-caleb')

    GCS_DATASET_DIR = KaggleDatasets().get_gcs_path(DATASET_DIR.parts[-1])

    MODEL_BUCKET = GCS_DATASET_DIR.split('/')[-1]

    PATH = Path('/kaggle/input/flower-classification-with-tpus')

    TFRECORD_DIR = KaggleDatasets().get_gcs_path(PATH.parts[-1])

    TPU_NAME = None

else:

    DATASET_DIR = Path('./flowers-caleb')

    MODEL_BUCKET = BUCKET

    PATH = Path('/home/jupyter/.fastai/data/flowers')

    TFRECORD_DIR = f'gs://{BUCKET}'

    TPU_NAME = 'dfdc-1'

    

SIZES = {s: f'{s}x{s}' for s in [192, 224, 331, 512]}



AUTO = tf.data.experimental.AUTOTUNE
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_NAME)

    print('Running on TPU ', tpu.master())

except:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

    

    # CONVERT DEGREES TO RADIANS

    pi = tf.constant(3.14159265359, tf.float32)

    rotation = pi * rotation / 180.

    shear = pi * shear / 180.

    

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

    

    return dot(dot(rotation_matrix, shear_matrix), dot(zoom_matrix, shift_matrix))
def transform(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = tf.shape(image)[0]

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

    idx2 = dot(m,tf.cast(idx,dtype='float32'))

    idx2 = tf.cast(idx2,dtype='int32')

    idx2 = tf.clip_by_value(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])
def augment(example):

    new_example = example.copy()

    image = transform(new_example['image'])

    image = tf.image.random_brightness(image, 0.3)

    image = tf.image.random_contrast(image, 0.9, 1.1)

    image = tf.image.random_hue(image, 0.1)

    image = tf.image.random_jpeg_quality(image, 70, 100)

    image = tf.image.random_saturation(image, 0.95, 1.05)

    new_example['image'] = image

    del example

    

    return new_example



def get_preprocess_fn(input_size=(224, 224), batch_size=128, norm=None, test=False):

    

    def imagenet_norm(image):

        mean = tf.constant([0.485, 0.456, 0.406])

        std = tf.constant([0.229, 0.224, 0.225])

        

        return (image / tf.constant(255, tf.float32) - mean) / std

    

    norm_fn = {'per_image': tf.image.per_image_standardization,

               'imagenet': imagenet_norm,

               None: tf.image.per_image_standardization

              }



    def preprocess(batch):

        image = tf.image.resize(batch['image'], input_size)

        image = norm_fn[norm](image)



        if test:

            return image

        

        else:

            image = tf.reshape(image, (batch_size, *input_size, 3))

            label = tf.cast(batch['label'], tf.float32)

            label = tf.reshape(label, (batch_size,))

                

            return image, label

        

    return preprocess

    

CLASSES = tf.constant(pd.read_csv(DATASET_DIR/'classes.csv').values.squeeze(), tf.string)



def get_parse_fn(split):

    def parse_fn(example):

        features = {"image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

                    "id": tf.io.FixedLenFeature([], tf.string),

                    "class": tf.io.FixedLenFeature([], tf.int64)}

        

        if split == 'test':

            del features['class']

        

        example = tf.io.parse_single_example(example, features)

        example['image'] = tf.image.decode_jpeg(example['image'], channels=3)

        

        if split != 'test':

            example['label'] = tf.cast(example['class'], tf.int32)

            example['class'] = CLASSES[example['label']]

        return example



    return parse_fn



def get_ds(split, img_size=224, batch_size=128, shuffle=False):

    file_pat = f'{TFRECORD_DIR}/tfrecords-jpeg-{SIZES[img_size]}/{split}/*.tfrec'

    

    options = tf.data.Options()

    options.experimental_deterministic = not shuffle

    

    ds = (tf.data.Dataset.list_files(file_pat, shuffle=shuffle)

          .with_options(options)

          .interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)

          .map(get_parse_fn(split), num_parallel_calls=AUTO)

         )

    

    if split == 'train':

        ds = ds.repeat().map(augment, num_parallel_calls=AUTO).shuffle(2048)

    

    return ds.batch(batch_size).prefetch(AUTO)
ds = get_ds('val')
for b in ds.take(1):

    b=b

b_aug = tf.map_fn(augment, b)
show_images(b['image'].numpy(), b['class'].numpy().tolist(), hw=(2,2), rc=(2,8))

show_images(b_aug['image'].numpy(), b_aug['class'].numpy().tolist(), hw=(2,2), rc=(2,8))
img_size = 512 

input_size = (512, 512)

batch_size = 128

weights = 'imagenet'



ds_train = get_ds('train', img_size=img_size, batch_size=batch_size, shuffle=True)

ds_valid = get_ds('val', img_size=img_size, batch_size=batch_size)



preprocess = get_preprocess_fn(batch_size=batch_size,

                               input_size=input_size, norm=weights)



ds_train_fit = ds_train.map(preprocess, num_parallel_calls=AUTO)

ds_valid_fit = ds_valid.map(preprocess, num_parallel_calls=AUTO)
model_prefix = 'model_efnb6_512_01'

model_dir = f'gs://{MODEL_BUCKET}/{model_prefix}'

checkpoint_dir = f'{model_dir}/checkpoints'

checkpoint_fn = checkpoint_dir + '/' + 'cp-{epoch:04d}.ckpt'

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=model_dir, write_graph=False, profile_batch=0)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_fn, save_weights_only=True)
if False:

    for b in bucket.list_blobs(prefix=f'{model_prefix}/checkpoints/cp-0052.ckpt'):

        print(b.name)

        b.download_to_filename(DATASET_DIR/b.name)
if False:

    for p in (DATASET_DIR/model_prefix/'checkpoints').glob('cp-0049*'):

        p.unlink()
cp_to_load = f'{checkpoint_dir}/cp-0052.ckpt'



with strategy.scope():

    

    opt = tf.keras.optimizers.Adam(1e-5)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    

    cnn = efn.EfficientNetB6(weights=None,include_top=False,pooling='avg', input_shape=(*input_size, 3))

    

#     for l in cnn.layers[:-32]:

#         l.trainable = False

    cnn.trainable = True



    model = tf.keras.Sequential([

        cnn,

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='selu', kernel_initializer="lecun_normal"),

        tf.keras.layers.AlphaDropout(0.5),

        tf.keras.layers.Dense(512, activation='selu', kernel_initializer="lecun_normal"),

        tf.keras.layers.AlphaDropout(0.5),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    

    if cp_to_load is not None:

        model.load_weights(cp_to_load)

    

    model.compile(loss=loss_fn, optimizer=opt, metrics=metrics)

    

model.summary()
for split in ['train', 'val', 'test']:

    items = 0

    for b in bucket.list_blobs(prefix=f'tfrecords-jpeg-{SIZES[img_size]}/{split}'):

        items += int(b.name.split('.')[0][-3:])

#         print(b.name)

    print(split, items, items // batch_size)
model.optimizer.learning_rate = 1e-4
if False:

    history = model.fit(ds_train_fit,

                        steps_per_epoch=200,

                        epochs=57,

                        initial_epoch=52,

                        validation_data=ds_valid_fit,

                        validation_steps=29,

                        callbacks=[checkpoint_cb, tensorboard_cb]

                       )
split = 'test'



ds_pred = get_ds(split, img_size=img_size, batch_size=batch_size)



preprocess = get_preprocess_fn(batch_size=batch_size,

                               input_size=input_size, norm=weights, test=(split == 'test'))



ds_pred_pp = ds_pred.map(preprocess, num_parallel_calls=AUTO)
# make sure example order is deterministic so we can line up training data with predictions

assert np.array_equal(np.concatenate([b['id'] for b in ds_pred.as_numpy_iterator()]),

               np.concatenate([b['id'] for b in ds_pred.as_numpy_iterator()]))
id_list = []

img_list = []

label_list = []

class_list = []



# TTA = 2

# if TTA is not None:

#     predictions = []

#     for b in tqdm(ds_pred.take(1)):

#         id_list.extend(b['id'].numpy().squeeze())

#         if split == 'val':

#             label_list.extend(b['label'].numpy().squeeze())

#             class_list.extend(b['class'].numpy().squeeze())

#         avg_preds = []

#         for i in range(TTA):

#             b_aug = (tf.data.Dataset.from_tensors(b).unbatch()

#                      .map(augment, num_parallel_calls=AUTO).batch(batch_size)

#                      .map(preprocess, num_parallel_calls=AUTO))

#             preds = model.predict(b_aug)

#             avg_preds.append(preds)

#         predictions.extend(np.mean(np.stack(avg_preds), axis=0))

#     predictions = np.stack(predictions, axis=0)

# else:



predictions = model.predict(ds_pred_pp)

for b in ds_pred.as_numpy_iterator():

    id_list.extend(b['id'].squeeze())

    img_list.extend(b['image'].squeeze())

    if split == 'val':

        label_list.extend(b['label'].squeeze())

        class_list.extend(b['class'].squeeze())
df_pred = pd.DataFrame({'id': [n.decode() for n in id_list]})



df_pred['label'] = np.argmax(predictions, axis=1)

df_pred['class'] = [n.decode() for n in np.tile(np.expand_dims(CLASSES.numpy(), axis=0),

                                (len(df_pred.label),))[:,df_pred.label].squeeze()]

df_pred['pred_prob'] = np.take_along_axis(predictions, np.expand_dims(df_pred.label, axis=1), axis=1)

    

if split == 'val':

    df_pred['actual_class'] = [n.decode() for n in class_list]

    df_pred['actual_label'] = label_list

    if len(img_list) > 0:

        df_pred['image'] = img_list

    

df_pred[['id', 'label']].to_csv('submission.csv', index=False)
if split == 'val':

    class_report = classification_report(df_pred.actual_label, df_pred.label, output_dict=True)

    df_cl_rep = pd.DataFrame(class_report).T.iloc[:103]

    df_cl_rep['f1-error'] = (1 - df_cl_rep['f1-score']) * df_cl_rep.support

    df_cl_rep = df_cl_rep.sort_values('f1-error', ascending=False)

    df_pred_g = pd.DataFrame(df_pred.groupby(['actual_label', 'label']).count()['id'])

    print(df_cl_rep.head(10))

    

    error_label = int(df_cl_rep.index[0])

    df_errors = df_pred[(df_pred.actual_label == error_label) & (df_pred.label != error_label)].copy()

    df_errors['n_class_err'] = df_errors.label.map(df_errors.groupby('label').count()['id'])

    if len(img_list) > 0:

        df_errors['image'] = df_errors.id.map(df_pred.set_index('id').image)

    df_errors = df_errors.sort_values(['n_class_err', 'label'], ascending=[False, False])

    print('\n',df_errors[[c for c in df_errors.columns if c != 'image']])

    

    show_images(df_errors.image.iloc[:16].to_list(),

                df_errors['class'].iloc[:16].to_list(),

                hw=(2,2), rc=(2,8))
if False:

    if not KAGGLE:

        print_output(run_command(f'kaggle d version -r tar -p {DATASET_DIR} -m "add model checkpoint"'))
%%javascript

IPython.notebook.save_notebook()
if True:

    if not KAGGLE:



        data = {'id': 'calebeverett/efficientnetb6-with-transformation',

                      'title': 'EfficientnetB6 with Transformation',

                      'code_file': 'flowers.ipynb',

                      'language': 'python',

                      'kernel_type': 'notebook',

                      'is_private': 'false',

                      'enable_gpu': 'true',

                      'enable_internet': 'true',

                      'dataset_sources': ['calebeverett/flowers-caleb'],

                      'competition_sources': ['flower-classification-with-tpus'],

                     ' kernel_sources': []}

        

        with open('kernel-metadata.json', 'w') as f:

            json.dump(data, f)



        print_output(run_command('kaggle k push'))