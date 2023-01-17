import math, re, os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets #Specific to Kaggle notebook environment
from tensorflow import keras
print(tf.__version__)
print(np.__version__)
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
files = {'192': {'train': [], 'val': [], 'test': []}, '224':{'train': [], 'val': [], 'test': []}, '331': {'train': [], 'val': [], 'test': []}, '512': {'train': [], 'val': [], 'test': []}}
try:
    data_loc = KaggleDatasets().get_gcs_path() #Built-in GCP Bucket in Kaggle for Data
except:
    data_loc = '/kaggle/input/flower-classification-with-tpus'
    
!gsutil ls $data_loc

for k in files.keys():
    subdir = f'/tfrecords-jpeg-{k}x{k}'
    loc = data_loc+subdir
    #print(loc)
    files[k]['train'] = tf.io.gfile.glob(loc + '/train/*.tfrec')
    files[k]['test'] = tf.io.gfile.glob(loc + '/test/*.tfrec')
    files[k]['val'] = tf.io.gfile.glob(loc + '/val/*.tfrec')
        
print(files['192'])
IMAGE_SIZE = [192, 192]

CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']

def readTrainVal(ex):
    LABELED_TFREC_FORMAT_TRVAL = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64)  # shape [] means single element
    }
    ex = tf.io.parse_single_example(ex, LABELED_TFREC_FORMAT_TRVAL)
    image_data = ex['image']
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) 
    label = tf.cast(ex['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def readTst(ex):
    LABELED_TFREC_FORMAT_TEST = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string)
    }
    ex = tf.io.parse_single_example(ex, LABELED_TFREC_FORMAT_TEST)
    image_data = ex['image']
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) 
    return image, ex['id'] # returns a dataset of (image, id) pairs

ds_test = tf.data.TFRecordDataset(files['192']['train'][0])
for d_rec in ds_test.take(1):
    i,l = readTrainVal(d_rec)
    plt.imshow(i)
    print(CLASSES[l])
ds_test_2 = tf.data.TFRecordDataset(files['224']['train'][0])
IMAGE_SIZE = [224, 224]
for d_rec in ds_test_2.take(1):
    i,l = readTrainVal(d_rec)
    plt.imshow(i)
    print(CLASSES[l])
    
ds_test_3 = tf.data.TFRecordDataset(files['331']['train'][0])
IMAGE_SIZE = [331, 331]
for d_rec in ds_test_3.take(1):
    i,l = readTrainVal(d_rec)
    plt.imshow(i)
    print(CLASSES[l])
    
ds_test_4 = tf.data.TFRecordDataset(files['512']['train'][0])
IMAGE_SIZE = [512, 512]
for d_rec in ds_test_4.take(1):
    i,l = readTrainVal(d_rec)
    plt.imshow(i)
    print(CLASSES[l])
    
IMAGE_SIZE = [192, 192]
num_img = {'test': 462, 'train': 798, 'val': 232}
AUTO = tf.data.experimental.AUTOTUNE

def training_dataset(f=None, batch_factor=2):
    d_train = tf.data.TFRecordDataset(files[str(IMAGE_SIZE[0])]['train'])
    d_train = d_train.map(readTrainVal)
    if f != None:
        d_train = d_train.map(f)
    b_size =  num_img['train'] // batch_factor
    d_train = d_train.shuffle(2048)
    d_train = d_train.batch(b_size)
    d_train = d_train.prefetch(AUTO)
    return d_train

def validation_dataset(batch_factor=2):
    d_val = tf.data.TFRecordDataset(files[str(IMAGE_SIZE[0])]['val'])
    d_val = d_val.map(readTrainVal)
    b_size = num_img['val'] // batch_factor
    d_val = d_val.shuffle(2048)
    d_val = d_val.batch(b_size)
    d_val = d_val.prefetch(AUTO)
    return d_val

def testing_dataset(batch_factor=2):
    d_tst = tf.data.TFRecordDataset(files[str(IMAGE_SIZE[0])]['test'])
    d_tst = d_tst.map(readTst)
    b_size =  num_img['test'] // batch_factor
    d_tst = d_tst.shuffle(2048)
    d_tst = d_tst.batch(b_size)
    d_tst = d_tst.prefetch(AUTO)
    return d_tst
    
    
tr = training_dataset()
val = validation_dataset()
tst = testing_dataset()
print("Training data shapes:")
for image, label in tr.take(-1):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy()[:10])
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
#Demo model - will need to run convolutional layers on tpus

with strategy.scope(): #This is for determining if the model runs on the TPU
    mod_conv = keras.Sequential()
    mod_conv.add(Conv2D(32, kernel_size=3, padding='same' ,activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    mod_conv.add(MaxPooling2D(pool_size=(2, 2)))
    mod_conv.add(Flatten())
    mod_conv.add(Dense(512, activation='relu'))
    mod_conv.add(Dense(104,activation='softmax'))
    o = keras.optimizers.SGD(learning_rate = 0.05, momentum=0.3)
    mod_conv.compile(optimizer=o, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mod_conv.summary()
mod_conv.fit(tr, epochs=5, validation_data=val) #Too much to run on cpu
augs = []
for d_rec in ds_test.take(1):
    i, l = readTrainVal(d_rec)
    plt.imshow(i)
    augs.append(tf.image.random_brightness(i, 0.8))
    augs.append(tf.image.random_contrast(i, 0.2, 0.8))
    augs.append(tf.image.random_flip_left_right(i))
    augs.append(tf.image.random_flip_up_down(i))
    augs.append(tf.image.random_hue(i, 0.3))
    augs.append(tf.image.random_jpeg_quality(i, 30, 95))
    augs.append(tf.image.random_saturation(i, 0.1, 0.8))
    augs.append(tf.image.central_crop(i, central_fraction=np.random.rand()))
    augs.append(tf.image.rot90(i, k=np.random.randint(4)))

for i in augs:
    fig, ax = plt.subplots()
    plt.imshow(i)

def augment(img, lab):
    ch = np.random.rand()
    if ch <= 0.6:
        return img, lab #keep ~60% of images in batch same
    n_apply = np.random.randint(9)
    make_trans = np.random.randint(9, size=n_apply)
    for m in make_trans:
        if m == 0:
            img = tf.image.random_brightness(img, 0.8)
        elif m == 1:
            img = tf.image.random_contrast(img, 0.2, 0.8)
        elif m == 2:
            img = tf.image.random_flip_left_right(img)
        elif m == 3:
            img = tf.image.random_flip_up_down(img)
        elif m == 4:
            img = tf.image.random_hue(img, 0.3)
        elif m == 5: 
            img = tf.image.random_jpeg_quality(img, 30, 95)
        elif m == 6:
            img = tf.image.random_saturation(img, 0.1, 0.8)
        elif m == 7:
            img = tf.image.central_crop(img, central_fraction=np.random.rand())
        else:
            img = tf.image.rot90(img, k=np.random.randint(4))
    return img, lab

tr = training_dataset(f=augment)
for d_rec in ds_test.take(1):
    i, l = readTrainVal(d_rec)
    i, l = augment(i,l)
    plt.imshow(i)
mod_conv.fit(tr, epochs=5, validation_data=val)
for image, label in tr.take(1): #Get first batch of images
    for i in range(10): #take 10 images from first batch
        fig, ax = plt.subplots()
        plt.title("train")
        plt.imshow(image[i])
        
for image, label in val.take(1): 
    for i in range(10): 
        fig, ax = plt.subplots()
        plt.title("val")
        plt.imshow(image[i])
with strategy.scope(): #This is for determining if the model runs on the TPU
    mod_conv_2 = keras.Sequential()
    mod_conv_2.add(Conv2D(32, kernel_size=25, strides=3,padding='same' ,activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    mod_conv_2.add(MaxPooling2D(pool_size=(6, 6))) 
    mod_conv_2.add(Conv2D(64, kernel_size=11,padding='same' ,activation='relu'))
    mod_conv_2.add(MaxPooling2D(pool_size=(2, 2))) 
    mod_conv_2.add(Conv2D(128, kernel_size=3 ,padding='same' ,activation='relu'))
    mod_conv_2.add(MaxPooling2D(pool_size=(2, 2))) 
    mod_conv_2.add(Flatten())
    mod_conv_2.add(Dense(512, activation='relu'))
    mod_conv_2.add(Dense(104,activation='softmax'))
    o = keras.optimizers.SGD(learning_rate = 0.05, momentum=0.3)
    mod_conv_2.compile(optimizer=o, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mod_conv_2.summary()
history = mod_conv_2.fit(tr, epochs=30, validation_data=val)
def plotLoss(ep, h):
    fig, ax = plt.subplots()
    plt.plot(range(ep), h.history['loss'], label='training loss')
    plt.plot(range(ep), h.history['val_loss'], label='validation loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
def plotAcc(ep, h):
    fig, ax = plt.subplots()
    plt.plot(range(ep), h.history['accuracy'], label='accuracy')
    plt.plot(range(ep), h.history['val_accuracy'], label='validation accuracy')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
plotLoss(30, history)
plotAcc(30, history)
with strategy.scope(): #This is for determining if the model runs on the TPU
    mod_conv_3 = keras.Sequential()
    mod_conv_3.add(Conv2D(32, kernel_size=11,padding='same' ,activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    mod_conv_3.add(MaxPooling2D(pool_size=(2, 2))) 
    mod_conv_3.add(Conv2D(64, kernel_size=3 ,padding='same' ,activation='relu'))
    mod_conv_3.add(MaxPooling2D(pool_size=(2, 2))) 
    mod_conv_3.add(Flatten())
    mod_conv_3.add(Dense(512, activation='relu'))
    mod_conv_3.add(Dense(104,activation='softmax'))
    o = keras.optimizers.SGD(learning_rate = 0.05, momentum=0.3)
    mod_conv_3.compile(optimizer=o, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mod_conv_3.summary()
history = mod_conv_3.fit(tr, epochs=30, validation_data=val)
plotLoss(30, history)
plotAcc(30, history)
with strategy.scope(): #This is for determining if the model runs on the TPU
    mod_conv_4 = keras.Sequential()
    mod_conv_4.add(Conv2D(32, kernel_size=11,padding='same' ,activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    mod_conv_4.add(Conv2D(32, kernel_size=11 ,activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    mod_conv_4.add(MaxPooling2D(pool_size=(2, 2))) 
    mod_conv_4.add(Conv2D(64, kernel_size=3 ,padding='same' ,activation='relu'))
    mod_conv_4.add(Conv2D(64, kernel_size=3,activation='relu'))
    mod_conv_4.add(MaxPooling2D(pool_size=(2, 2))) 
    mod_conv_4.add(Flatten())
    mod_conv_4.add(Dense(2048, activation='relu'))
    mod_conv_4.add(Dense(104,activation='softmax'))
    o = keras.optimizers.SGD(learning_rate = 0.05, momentum=0.3)
    mod_conv_4.compile(optimizer=o, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mod_conv_4.summary()
history = mod_conv_4.fit(tr, epochs=30, validation_data=val)
plotLoss(30, history)
plotAcc(30, history)
def plotSparse(ep, h):
    fig, ax = plt.subplots()
    plt.plot(range(ep), h.history['sparse_categorical_accuracy'], label='accuracy')
    plt.plot(range(ep), h.history['val_sparse_categorical_accuracy'], label='validation accuracy')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
with strategy.scope():
    mod_conv_5 = keras.models.clone_model(mod_conv_3)
    o = keras.optimizers.SGD(learning_rate = 0.05, momentum=0.3)
    mod_conv_5.compile(optimizer=o, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
mod_conv_5.summary()
history = mod_conv_5.fit(tr, epochs=30, validation_data=val)
plotLoss(30, history)
plotSparse(30, history)
with strategy.scope():
    mod_conv_6 = keras.models.clone_model(mod_conv_4)
    o = keras.optimizers.SGD(learning_rate = 0.05, momentum=0.3)
    mod_conv_6.compile(optimizer=o, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
mod_conv_6.summary()
history = mod_conv_6.fit(tr, epochs=30, validation_data=val)
plotLoss(30, history)
plotSparse(30, history)
history = mod_conv_6.fit(tr, epochs=20, validation_data=val)
plotLoss(20, history)
plotSparse(20, history)
with strategy.scope():
    pt = keras.applications.resnet.ResNet50(weights='imagenet', input_shape=[*IMAGE_SIZE, 3], include_top=False)
    mod_pt_1 = keras.Sequential()
    mod_pt_1.add(pt)
    mod_pt_1.add(keras.layers.GlobalAveragePooling2D()) #using this to get average of large datasets
    mod_pt_1.add(Dense(2048, activation='relu'))
    mod_pt_1.add(Dense(104,activation='softmax'))
    o = keras.optimizers.SGD(learning_rate = 0.05, momentum=0.3)
    mod_pt_1.compile(optimizer=o, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
mod_pt_1.summary()
history = mod_pt_1.fit(tr, epochs=30, validation_data=val)
plotLoss(30, history)
plotSparse(30, history)
with strategy.scope():
    pt = keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=[*IMAGE_SIZE, 3], include_top=False)
    mod_pt_2 = keras.Sequential()
    mod_pt_2.add(pt)
    mod_pt_2.add(keras.layers.GlobalAveragePooling2D()) #using this to get average of large datasets
    mod_pt_2.add(Dense(2048, activation='relu'))
    mod_pt_2.add(Dense(104,activation='softmax'))
    o = keras.optimizers.SGD(learning_rate = 0.05, momentum=0.3)
    mod_pt_2.compile(optimizer=o, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
mod_pt_2.summary()
history = mod_pt_2.fit(tr, epochs=30, validation_data=val)
plotLoss(30, history)
plotSparse(30, history)
opt = [keras.optimizers.SGD(momentum=0.7), keras.optimizers.SGD(learning_rate = 0.7), 'adagrad', 'adam']
hist = []
for o in opt:
    with strategy.scope():
        pt = keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=[*IMAGE_SIZE, 3], include_top=False)
        m = keras.Sequential()
        m.add(pt)
        m.add(keras.layers.GlobalAveragePooling2D()) #using this to get average of large datasets
        m.add(Dense(2048, activation='relu'))
        m.add(Dense(104,activation='softmax'))
        m.compile(optimizer=o, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    history = m.fit(tr, epochs=30, validation_data=val)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    hist.append(history)
print(len(hist))
for i,h in enumerate(hist):
    if i == 0:
        print('High Momentum')
    elif i == 1:
        print('High learning rate')
    elif i == 2:
        print('Adagrad')
    else:
        print('Adam')
    plotLoss(30, h)
    plotSparse(30, h)
with strategy.scope():
    pt = keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=[*IMAGE_SIZE, 3], include_top=False)
    best_model = keras.Sequential()
    best_model.add(pt)
    best_model.add(keras.layers.GlobalAveragePooling2D()) #using this to get average of large datasets
    best_model.add(Dense(2048, activation='relu'))
    best_model.add(Dense(104,activation='softmax'))
    best_model.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

best_model.summary()
tr_2 = training_dataset(batch_factor=3)
val_2 = validation_dataset(batch_factor=3)
bt_1 = keras.models.clone_model(best_model)
bt_1.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
h = bt_1.fit(tr_2, epochs=20, validation_data=val_2)
plotLoss(20, h)
plotSparse(20, h)
with strategy.scope():
    pt = keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=[*IMAGE_SIZE, 3], include_top=False)
    final_model = keras.Sequential()
    final_model.add(pt)
    final_model.add(keras.layers.GlobalAveragePooling2D()) #using this to get average of large datasets
    final_model.add(Dropout(0.3))
    final_model.add(Dense(2048, activation='relu'))
    final_model.add(Dense(104,activation='softmax'))
    final_model.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

final_model.summary()
h = final_model.fit(tr, epochs=30, validation_data=val)
plotLoss(30, h)
plotSparse(30, h)
def export(model):
    pic_batch = tst.map(lambda image, idnum: image)
    p = model.predict(pic_batch)
    pred = np.argmax(p, axis=-1)
    i = 0
    for f in pic_batch.unbatch().take(10):
        fig, ax = plt.subplots()
        plt.title(CLASSES[pred[i]])
        plt.imshow(f)
        i += 1

export(final_model)
    
final_model.save('final_model.h5')