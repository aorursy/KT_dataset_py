!pip install -q quick-ml
import tensorflow as tf

import quick_ml
#DATA_DIR = '../input/iucn-animal-species/data'



DATA_DIR = '/kaggle/input/iucn-animal-species/data'
"""from quick_ml.tfrecords_maker import create_tfrecord_labeled



from quick_ml.tfrecords_maker import get_addrs_labels



addrs, labels = get_addrs_labels(DATA_DIR)"""
"""out_filename = 'train.tfrecords'



create_tfrecord_labeled(addrs, labels, out_filename, IMAGE_SIZE = (192,192), num_parts = 1)"""
from quick_ml.tfrecords_maker import create_split_tfrecords_data
outfile1name = 'training.tfrecords'

outfile2name = 'validation.tfrecords'

output1folder = 'train'

output2folder = 'val'



split_size_ratio = 0.8





create_split_tfrecords_data(DATA_DIR, outfile1name, output1folder, outfile2name, output2folder, split_size_ratio,num_parts1 = 1, num_parts2 = 1, IMAGE_SIZE = (192,192))
dictionary_labeled = "{'image' : tf.io.FixedLenFeature([], tf.string), 'label' : tf.io.FixedLenFeature([], tf.int64)}"

IMAGE_SIZE = "192,192"





from quick_ml.begin_tpu import get_labeled_tfrecord_format

get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)
from quick_ml.visualize_and_check_data import check_one_image_and_label, check_batch_and_labels, check_one_image_and_id, check_batch_and_ids
tfrecord_filename = '/kaggle/working/train/training.tfrecords'



n_examples = 9

grid_rows = 3

grid_columns = 3



check_batch_and_labels(tfrecord_filename, n_examples, grid_rows, grid_columns, grid_size = (10,10))
tfrecord_filename = '/kaggle/working/val/validation.tfrecords'



n_examples = 9

grid_rows = 3

grid_columns = 3



check_batch_and_labels(tfrecord_filename, n_examples, grid_rows, grid_columns, grid_size = (10,10))