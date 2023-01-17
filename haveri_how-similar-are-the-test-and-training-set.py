# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import math, re, gc

import numpy as np # linear algebra

import pickle

from datetime import datetime, timedelta

from multiprocessing import Pool

import tensorflow as tf

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print('TensorFlow version', tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE



import scipy.io

import random

import os

from datetime import datetime, timedelta

import tensorflow as tf



import skimage

from skimage.io import imread as SKImageRead

from skimage.io import imsave as SKImageSave

from skimage.util import crop as SKImageCrop

from skimage.transform import resize as SKImageResize

from skimage.metrics import structural_similarity as ssim

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



print('Replicas:', strategy.num_replicas_in_sync)



GCS_DS_PATH = KaggleDatasets().get_gcs_path('flower-classification-with-tpus')

MORE_IMAGES_GCS_DS_PATH = KaggleDatasets().get_gcs_path('tf-flower-photo-tfrec')

print(GCS_DS_PATH, '\n', MORE_IMAGES_GCS_DS_PATH)

!ls -ltr $GCS_DS_PATH $MORE_IMAGES_GCS_DS_PATH

GCS_DS_PATH = '/kaggle/input/flower-classification-with-tpus'

MORE_IMAGES_GCS_DS_PATH = '/kaggle/input/tf-flower-photo-tfrec'

print(GCS_DS_PATH, '\n', MORE_IMAGES_GCS_DS_PATH)

!cp /kaggle/input/efficientnet-with-all-5-imagesets-s1/submission.csv ./

#!ls -ltr ./

#!ls -l /kaggle/input/tf-flower-photo-tfrec/*/tfrecords-jpeg-224x224/*.tfrec

#!ls -l /kaggle/input/tf-flower-photo-tfrec/imagenet/tfrecords-jpeg-224x224/*.tfrec

#!ls -l /kaggle/input/tf-flower-photo-tfrec/inaturalist/tfrecords-jpeg-224x224/*.tfrec

#!ls -l /kaggle/input/tf-flower-photo-tfrec/openimage/tfrecords-jpeg-224x224/*.tfrec

#!ls -l /kaggle/input/tf-flower-photo-tfrec/oxford_102/tfrecords-jpeg-224x224/*.tfrec

#!ls -l /kaggle/input/tf-flower-photo-tfrec/tf_flowers/tfrecords-jpeg-224x224/*.tfrec
start_time = datetime.now()

print('Time now is', start_time)



IMAGE_SIZE = [224, 224] # [512, 512]



EPOCHS = 12

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



GCS_PATH_SELECT = {

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')



MOREIMAGES_PATH_SELECT = {

    192: '/tfrecords-jpeg-192x192',

    224: '/tfrecords-jpeg-224x224',

    331: '/tfrecords-jpeg-331x331',

    512: '/tfrecords-jpeg-512x512'

}

MOREIMAGES_PATH = MOREIMAGES_PATH_SELECT[IMAGE_SIZE[0]]



IMAGENET_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/imagenet' + MOREIMAGES_PATH + '/*.tfrec')

INATURELIST_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/inaturalist' + MOREIMAGES_PATH + '/*.tfrec')

OPENIMAGE_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/openimage' + MOREIMAGES_PATH + '/*.tfrec')

OXFORD_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/oxford_102' + MOREIMAGES_PATH + '/*.tfrec')

TENSORFLOW_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/tf_flowers' + MOREIMAGES_PATH + '/*.tfrec')

ADDITIONAL_TRAINING_FILENAMES = IMAGENET_FILES + INATURELIST_FILES + OPENIMAGE_FILES + OXFORD_FILES + TENSORFLOW_FILES

#print(TEST_FILENAMES)

print('----')

TRAINING_FILENAMES = TRAINING_FILENAMES + ADDITIONAL_TRAINING_FILENAMES

#print(TRAINING_FILENAMES)



# This is so awkward. Everyone is doing this for an extra few points.

# TRAINING_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES

# VALIDATION_FILENAMES = TRAINING_FILENAMES



CLASSES = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'wild geranium', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', # 00 - 09

           'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', # 10 - 19

           'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', # 20 - 29

           'carnation', 'garden phlox', 'love in the mist', 'cosmos', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', # 30 - 39

           'barberton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'daisy', 'common dandelion', # 40 - 49

           'petunia', 'wild pansy', 'primula', 'sunflower', 'lilac hibiscus', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia', # 50 - 59

           'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'iris', 'windflower', 'tree poppy', # 60 - 69

           'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', # 70 - 79

           'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', # 80 - 89

           'hippeastrum ', 'bee balm', 'pink quill', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', # 90 - 99

           'trumpet creeper', 'blackberry lily', 'common tulip', 'wild rose'] # 100 - 102
def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)

#



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

print('Dataset: {} training images, {} unlabeled test images. Total possible ssim comparisions {}'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES, (NUM_TRAINING_IMAGES * NUM_TEST_IMAGES)))

print('Dataset: {} training images, {} labeled validation images. Total possible ssim comparisions {}'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, (NUM_TRAINING_IMAGES * NUM_VALIDATION_IMAGES)))
print(len(TRAINING_FILENAMES), len(VALIDATION_FILENAMES), len(TEST_FILENAMES))
filename = './submission.csv'



my_submission = np.loadtxt(filename, dtype=str, skiprows=1, unpack=True)

image_predictions = {}

for aline in my_submission:

    splitstring = aline.split(',')

    image_predictions[splitstring[0]] = int(splitstring[1])

print(type(my_submission), my_submission.shape)

print(type(image_predictions))

#
def find_similar_flowers_in_test_train(id, image_predictions, flowers_test_files, flowers_training_files):

#    no_of_test_images = 0

    failure_threshold = 0.9

    total_comparisons = 0

    image_by_cls_id = {} # For each image we store a list of tuples (idnum, training_images_exists_in_test, image)

    for i in range(104):

        image_by_cls_id[i] = []

#

    def read_labeled_tfrecord(example):

        LABELED_TFREC_FORMAT = {

            'image': tf.io.FixedLenFeature([], tf.string),

            'class': tf.io.FixedLenFeature([], tf.int64),

        }

        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

        image = tf.image.decode_jpeg(example['image'])

        label = tf.cast(example['class'], tf.int32)

        return image, label

#

    def read_unlabeled_tfrecord(example):

        UNLABELED_TFREC_FORMAT = {

            'image': tf.io.FixedLenFeature([], tf.string),

            'id': tf.io.FixedLenFeature([], tf.string),

        }

        example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

        image = tf.image.decode_jpeg(example['image'])

        idnum = example['id']

        return image, idnum

#

    if image_predictions == None:

        for filename in flowers_test_files:

            raw_image_dataset = tf.data.TFRecordDataset(filename)

            image_dataset = raw_image_dataset.map(read_labeled_tfrecord)

            idnum_prefix = re.compile(r"\/([0-9].*)\.tfrec").search(filename).group(1) + '_img_'

#            print('filename is {}, idnum_prefix is {}'.format(filename, idnum_prefix))

            no_of_test_images = 0

            for image_features in image_dataset:

                idnum = idnum_prefix + str(no_of_test_images)

                no_of_test_images = no_of_test_images + 1

                image, cls_id = image_features

                image = image.numpy()

                cls_id = cls_id.numpy()

                curr_image_data = (idnum, 0, image)

                image_by_cls_id[cls_id].append(curr_image_data)

    else:

        for filename in flowers_test_files:

            raw_image_dataset = tf.data.TFRecordDataset(filename)

            image_dataset = raw_image_dataset.map(read_unlabeled_tfrecord)

            for image_features in image_dataset:

#                no_of_test_images = no_of_test_images + 1

                image, idnum = image_features

                image = image.numpy()

                idnum = idnum.numpy().decode('UTF-8')

                cls_id = image_predictions[idnum]

                curr_image_data = (idnum, 0, image)

                image_by_cls_id[cls_id].append(curr_image_data)

#

#    print(no_of_test_images)

#

    def check_ssim_with_test_images(filename, img_id_within_tfrec, trn_label, image):

        comparisons = 0 # TODO

        messages = []

        compare_test_images = image_by_cls_id[trn_label]

        no_of_compares = len(compare_test_images)

        for j in range(no_of_compares):

#            if j >= 20:

#                break

            curr_image_data = image_by_cls_id[trn_label][j]

            idnum, training_images_exists_in_test, test_image = curr_image_data

            if training_images_exists_in_test:

                continue

            comparisons = comparisons + 1

            ssim_val = ssim(test_image, image, multichannel=True)

            if ssim_val > failure_threshold:

                curr_image_data = (idnum, 1, test_image)

                image_by_cls_id[trn_label][j] = curr_image_data

                this_message = 'Image {} similar with image in training {} img number {}. similarity index is {}'.format(idnum, filename, img_id_within_tfrec, ssim_val)

                messages.append(this_message)

        return comparisons, messages

#

    print('Start {}: {}'.format(id, datetime.now()))

    all_messages = []

    for filename in flowers_training_files:

        raw_image_dataset = tf.data.TFRecordDataset(filename)

        image_dataset = raw_image_dataset.map(read_labeled_tfrecord)

        img_id_within_tfrec = 0

        for image_features in image_dataset:

#            if img_id_within_tfrec >= 20:

#                break

            image, label = image_features

            image = image.numpy()

            label = label.numpy()

            comparisons, messages = check_ssim_with_test_images(filename, img_id_within_tfrec, label, image)

            total_comparisons = total_comparisons + comparisons

            all_messages.extend(messages)

            img_id_within_tfrec = img_id_within_tfrec + 1

    #

    print('End {}: Total Comparisons {}: Images matched {}: {}'.format(id, total_comparisons, len(all_messages), datetime.now()))

    filename = str(id) + '_in_case_i_crash_timeout_similar_images.csv'

    np.savetxt(filename, np.rec.fromarrays([all_messages]), fmt = ['%s'], delimiter = ',', header = 'messages', comments = '')

    return all_messages

#
#flowers_test_files = TEST_FILENAMES

#flowers_training_files = TRAINING_FILENAMES

#find_similar_flowers_in_test_train(0, image_predictions, flowers_test_files, flowers_training_files)

all_messages = [] # setting up to prevent failing of this kernel to be published.
mul_proc_params = []

no_test_tfrec_files = len(TEST_FILENAMES)

shift_1 = len(TRAINING_FILENAMES) // no_test_tfrec_files

for i in range(no_test_tfrec_files):

    test_file = [TEST_FILENAMES[i]]

    shift = i * shift_1

    train_files = TRAINING_FILENAMES[shift:] + TRAINING_FILENAMES[:shift]

    mul_proc_params.append((i, image_predictions, test_file, train_files))

#

# mul_proc_params = mul_proc_params[12:]

#

def is_flower_similar():

    with Pool(6) as p:

        all_messages = p.starmap(find_similar_flowers_in_test_train, mul_proc_params)

        print(len(all_messages))

    return all_messages

#

#if __name__ == '__main__':

#    all_messages = is_flower_similar()

#
mul_proc_params = []

no_val_tfrec_files = len(VALIDATION_FILENAMES)

shift_1 = len(TRAINING_FILENAMES) // no_val_tfrec_files

for i in range(no_val_tfrec_files):

    val_file = [VALIDATION_FILENAMES[i]]

    shift = i * shift_1

    train_files = TRAINING_FILENAMES[shift:] + TRAINING_FILENAMES[:shift]

    mul_proc_params.append((i, None, val_file, train_files))

#

print(len(mul_proc_params))

mul_proc_params = mul_proc_params[8:]

#

def is_flower_similar():

    with Pool(5) as p:

        all_messages = p.starmap(find_similar_flowers_in_test_train, mul_proc_params)

        print(len(all_messages))

    return all_messages

#

#if __name__ == '__main__':

#    all_messages = is_flower_similar()

#
single_list_messages = []

for i in range(len(all_messages)):

    single_list_messages.extend(all_messages[i])

#

print('Number of images similar with training images', len(single_list_messages))

#
filename = 'similar_images.csv'

np.savetxt(filename, np.rec.fromarrays([single_list_messages]), fmt = ['%s'], delimiter = ',', header = 'messages', comments = '')

#
def visualize(image1, image1_title, image2, image2_title):

    image1_title = 'test id - ' + image1_title

    image2_title = 'ssi - ' + image2_title

    fig = plt.figure()

    plt.subplot(1,2,1)

    plt.title(image1_title)

    plt.imshow(image1)



    plt.subplot(1,2,2)

    plt.title(image2_title)

    plt.imshow(image2)

#
#

def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'class': tf.io.FixedLenFeature([], tf.int64),

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = tf.image.decode_jpeg(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label

#

def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'id': tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = tf.image.decode_jpeg(example['image'])

    idnum = example['id']

    return image, idnum

#

def get_training_images(filename):

    training_image = []

    raw_image_dataset = tf.data.TFRecordDataset(filename)

    image_dataset = raw_image_dataset.map(read_labeled_tfrecord)

    for image_features in image_dataset:

        image, label = image_features

        image = image.numpy()

        training_image.append(image)

    return training_image

#
test_images = {}

no_of_test_images = 0

#

for filename in TEST_FILENAMES:

    raw_image_dataset = tf.data.TFRecordDataset(filename)

    image_dataset = raw_image_dataset.map(read_unlabeled_tfrecord)

    for image_features in image_dataset:

        no_of_test_images = no_of_test_images + 1

#        if no_of_test_images > 2:

#            break

        image, idnum = image_features

        image = image.numpy()

        idnum = idnum.numpy().decode('UTF-8')

        test_images[idnum] = image

#
#

testids = test_images.keys()

print('Test', len(TEST_FILENAMES), no_of_test_images, len(testids))

#
#Image 61f976a63 similar to /tfrecords-jpeg-224x224/train/06-224x224-798.tfrec img number 353. similarity index is 1.0

#Image 9eac41e56 similar to /tfrecords-jpeg-224x224/train/06-224x224-798.tfrec img number 546. similarity index is 1.0

#Image 805888e57 similar to /tfrecords-jpeg-224x224/train/06-224x224-798.tfrec img number 627. similarity index is 1.0

#

test_ids_to_show = ['61f976a63', '9eac41e56', '805888e57']

ids_similar_to = [353, 546, 627]

ssi_vals = [1.0, 1.0, 1.0]

filename = GCS_DS_PATH + '/tfrecords-jpeg-224x224/train/06-224x224-798.tfrec'

training_images = get_training_images(filename)



visualize(test_images[test_ids_to_show[0]], test_ids_to_show[0], training_images[ids_similar_to[0]], str(ssi_vals[0]))

visualize(test_images[test_ids_to_show[1]], test_ids_to_show[1], training_images[ids_similar_to[1]], str(ssi_vals[1]))

visualize(test_images[test_ids_to_show[2]], test_ids_to_show[2], training_images[ids_similar_to[2]], str(ssi_vals[2]))

#
#Image 3bef19347 similar to /tf_flowers/tfrecords-jpeg-224x224/7-224x224-99.tfrec img number 79. similarity index is 0.9631336470049914

#Image c9e27d0e3 similar to /tf_flowers/tfrecords-jpeg-224x224/7-224x224-99.tfrec img number 88. similarity index is 0.9579509019834677

#Image 49292d94c similar to /tf_flowers/tfrecords-jpeg-224x224/7-224x224-99.tfrec img number 91. similarity index is 0.963339089650537

#

test_ids_to_show = ['3bef19347', 'c9e27d0e3', '49292d94c']

ids_similar_to = [79, 88, 91]

ssi_vals = [0.9631, 0.9579, 0.9633]

filename = MORE_IMAGES_GCS_DS_PATH + '/tf_flowers/tfrecords-jpeg-224x224/7-224x224-99.tfrec'

training_images = get_training_images(filename)



visualize(test_images[test_ids_to_show[0]], test_ids_to_show[0], training_images[ids_similar_to[0]], str(ssi_vals[0]))

visualize(test_images[test_ids_to_show[1]], test_ids_to_show[1], training_images[ids_similar_to[1]], str(ssi_vals[1]))

visualize(test_images[test_ids_to_show[2]], test_ids_to_show[2], training_images[ids_similar_to[2]], str(ssi_vals[2]))

#
#Image a0925df64 similar to /imagenet/tfrecords-jpeg-224x224/0-224x224-1852.tfrec img number 142. similarity index is 0.9171391656692433

#Image 360538bb1 similar to /imagenet/tfrecords-jpeg-224x224/0-224x224-1852.tfrec img number 809. similarity index is 0.9009493950335353

#Image 8482cd4e5 similar to /imagenet/tfrecords-jpeg-224x224/0-224x224-1852.tfrec img number 1268. similarity index is 0.9017363495863923

#

test_ids_to_show = ['a0925df64', '360538bb1', '8482cd4e5']

ids_similar_to = [142, 809, 1268]

ssi_vals = [0.9171, 0.9009, 0.9017]

filename = MORE_IMAGES_GCS_DS_PATH + '/imagenet/tfrecords-jpeg-224x224/0-224x224-1852.tfrec'

training_images = get_training_images(filename)



visualize(test_images[test_ids_to_show[0]], test_ids_to_show[0], training_images[ids_similar_to[0]], str(ssi_vals[0]))

visualize(test_images[test_ids_to_show[1]], test_ids_to_show[1], training_images[ids_similar_to[1]], str(ssi_vals[1]))

visualize(test_images[test_ids_to_show[2]], test_ids_to_show[2], training_images[ids_similar_to[2]], str(ssi_vals[2]))

#