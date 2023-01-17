import math, re, os

import tensorflow as tf

from tensorflow.keras import backend as K

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print("Tensorflow version " + tf.__version__)
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



print("REPLICAS: ", strategy.num_replicas_in_sync)
HEIGHT = 512

WIDTH = 512

CHANNELS = 3

N_CLASSES = 104

SHOW_LIMIT = 10

seed = 27



GCS_PATH = KaggleDatasets().get_gcs_path() + '/tfrecords-jpeg-%sx%s' % (HEIGHT, WIDTH)



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')



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
def batch_to_numpy_images_and_labels(data):

    images, labels = data

    

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    # Not showing labels so no need of if condition

    #if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

    numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels
def showimage(image1,image2,title1= 'Before augmentation',title2 ='After augmentation'):

    plt.figure(figsize=(15, 15))

    plt.subplot(121)

    plt.imshow(image1)

    plt.title(title1)

    

    plt.subplot(122)

    plt.imshow(image2)

    plt.title(title2)

    plt.show()
# Datasets utility functions

AUTO = tf.data.experimental.AUTOTUNE # instructs the API to read from multiple files if available.



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [HEIGHT, WIDTH, 3])

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label



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

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    image = tf.image.random_flip_left_right(image, seed=seed)

    image = tf.image.random_flip_up_down(image, seed=seed)

    image = tf.image.random_saturation(image, lower=0, upper=2, seed=seed)

    image = tf.image.random_contrast(image, lower=.8, upper=2, seed=seed)

    image = tf.image.random_brightness(image, max_delta=.2, seed=seed)

    image = tf.image.random_crop(image, size=[int(HEIGHT*.8), int(WIDTH*.8), CHANNELS], seed=seed)

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

    DIM = HEIGHT

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

dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

dataset_aug = dataset.map(transform, num_parallel_calls=AUTO)

for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2, label2 = batch_to_numpy_images_and_labels(data_aug)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
print('random_flip_left_right')

def random_flip_left_right(image, label):

    image = tf.image.random_flip_left_right(image, seed=seed)

    return image, label



dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

dataset_aug = dataset.map(random_flip_left_right, num_parallel_calls=AUTO)

for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2, label2 = batch_to_numpy_images_and_labels(data_aug)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
print('random_flip_up_down')

def random_flip_up_down(image, label):

    image = tf.image.random_flip_up_down(image, seed=seed)

    return image, label





dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

dataset_aug = dataset.map(random_flip_up_down, num_parallel_calls=AUTO)

for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2, label2 = batch_to_numpy_images_and_labels(data_aug)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
print('random_saturation')

def random_saturation(image, label):

    image = tf.image.random_saturation(image, lower=0, upper=2, seed=seed)

    return image, label



dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

dataset_aug = dataset.map(random_saturation, num_parallel_calls=AUTO)

for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2, label2 = batch_to_numpy_images_and_labels(data_aug)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
print('random_crop')

def random_crop(image, label):

    image = tf.image.random_crop(image, size=[int(HEIGHT*.8), int(WIDTH*.8), CHANNELS], seed=seed)

    return image, label



dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

dataset_aug = dataset.map(random_crop, num_parallel_calls=AUTO)

for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2, label2 = batch_to_numpy_images_and_labels(data_aug)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
print('random_contrast')

def random_contrast(image, label):

    image = tf.image.random_contrast(image, lower=.8, upper=2, seed=seed)

    return image, label



dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

dataset_aug = dataset.map(random_contrast, num_parallel_calls=AUTO)

for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2, label2 = batch_to_numpy_images_and_labels(data_aug)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
print('random_brightness')

def random_brightness(image, label):

    image = tf.image.random_brightness(image, max_delta=.2, seed=seed)

    return image, label



dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

dataset_aug = dataset.map(random_brightness, num_parallel_calls=AUTO)

for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2, label2 = batch_to_numpy_images_and_labels(data_aug)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
print('random_brightness')

def random_brightness(image, label):

    image = tf.image.random_brightness(image, max_delta=.2, seed=seed)

    return image, label



dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

dataset_aug = dataset.map(random_brightness, num_parallel_calls=AUTO)

for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2, label2 = batch_to_numpy_images_and_labels(data_aug)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
print('All together')

dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

dataset_aug = dataset.map(data_augment, num_parallel_calls=AUTO)

for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2, label2 = batch_to_numpy_images_and_labels(data_aug)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
import numpy as np

import cv2

def order_points(pts):

    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)

    rect[0] = pts[np.argmin(s)]

    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)

    rect[1] = pts[np.argmin(diff)]

    rect[3] = pts[np.argmax(diff)]

    return rect
def four_point_transform(image,label):

    cords = "[(0, 0), (0, 512), (512, 400), (512, 112)]"

    pts = np.array(eval(cords), dtype = "float32")

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([

        [0, 0],

        [maxWidth - 1, 0],

        [maxWidth - 1, maxHeight - 1],

        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

for i,data in enumerate(dataset):

    image1, label1 = batch_to_numpy_images_and_labels(data)

    image2 = four_point_transform(image1,None)

    showimage(image1,image2)

    if i == SHOW_LIMIT:

        break
## this will not work directly will have to keep everthing in tf

# print('four_point_transform')

# dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

# dataset_aug = dataset.map(four_point_transform)

# for i,(data, data_aug) in enumerate(zip(dataset,dataset_aug)):

#     image1, label1 = batch_to_numpy_images_and_labels(data)

#     image2, label2 = batch_to_numpy_images_and_labels(data_aug)

#     showimage(image1,image2)

#     if i == SHOW_LIMIT:

#         break