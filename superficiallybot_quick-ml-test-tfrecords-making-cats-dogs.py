!pip install -q quick-ml
import tensorflow as tf

import quick_ml
!unzip /kaggle/input/dogs-vs-cats/test1.zip
TEST_DIR = '/kaggle/working/test1'
from quick_ml.tfrecords_maker import create_tfrecord_unlabeled



from quick_ml.tfrecords_maker import get_addrs_ids
addrs, ids = get_addrs_ids(TEST_DIR)
unlabeled_dataset = create_tfrecord_unlabeled('test_cats_dogs_192x192.tfrecords', addrs, ids, IMAGE_SIZE = (192,192))
from quick_ml.visualize_and_check_data import check_batch_and_ids
dictionary_unlabeled = "{ 'image' : tf.io.FixedLenFeature([], tf.string), 'idnum' : tf.io.FixedLenFeature([], tf.string) }"

IMAGE_SIZE = "192,192"



from quick_ml.begin_tpu import get_unlabeled_tfrecord_format



get_unlabeled_tfrecord_format(dictionary_unlabeled, IMAGE_SIZE)
check_batch_and_ids('/kaggle/working/test_cats_dogs_192x192.tfrecords', 15, 3,5, (10,10))