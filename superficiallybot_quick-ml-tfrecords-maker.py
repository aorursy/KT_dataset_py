!pip install quick-ml
import tensorflow as tf
import quick_ml
! wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip -O catsdogs.zip
!unzip /kaggle/working/catsdogs.zip
!rm catsdogs.zip
from quick_ml.tfrecords_maker import create_tfrecord_labeled
from quick_ml.tfrecords_maker import get_addrs_labels
DATA_DIR = '/kaggle/working/PetImages'
addrs, labels = get_addrs_labels(DATA_DIR)
output_filename = 'train.tfrecords'

create_tfrecord_labeled(addrs, labels, output_filename, IMAGE_SIZE = (192,192))
from quick_ml.tfrecords_maker import create_split_tfrecords_data
outfile1name = 'training.tfrecords'

outfile2name = 'validation.tfrecords'



#create_split_tfrecords_data(DATA_DIR, outfile1name, outfile2name, split_size_ratio = 0.7, IMAGE_SIZE = (192,192))
from quick_ml.tfrecords_maker import create_tfrecord_unlabeled

from quick_ml.tfrecords_maker import get_addrs_ids
Unlabeled_Data_Dir = '/kaggle/working/PetImages/Cat'

addrs, ids = get_addrs_ids(Unlabeled_Data_Dir)
out_filename = 'unlabeled.tfrecords'

create_tfrecord_unlabeled(out_filename, addrs, ids, IMAGE_SIZE = (192,192))
from quick_ml.visualize_and_check_data import check_one_image_and_label
from quick_ml.begin_tpu import get_labeled_tfrecord_format



dictionary_labeled = "{ 'image' : tf.io.FixedLenFeature([], tf.string), 'label' : tf.io.FixedLenFeature([], tf.int64) }"

IMAGE_SIZE = "192,192"



get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)
tfrecord_filename = '/kaggle/working/train.tfrecords'





check_one_image_and_label(tfrecord_filename)
from quick_ml.visualize_and_check_data import check_batch_and_labels
tfrecord_filename = '/kaggle/working/train.tfrecords'

n_examples = 15

grid_rows = 3

grid_columns = 5

grid_size = (10,10)





check_batch_and_labels(tfrecord_filename, n_examples, grid_rows, grid_columns, grid_size)
from quick_ml.visualize_and_check_data import check_one_image_and_id
from quick_ml.begin_tpu import get_unlabeled_tfrecord_format



dictionary_unlabeled = "{ 'image' : tf.io.FixedLenFeature([], tf.string), 'idnum' : tf.io.FixedLenFeature([], tf.string) }"

IMAGE_SIZE = "192,192"



get_unlabeled_tfrecord_format(dictionary_unlabeled, IMAGE_SIZE)
tfrecord_filename = '/kaggle/working/unlabeled.tfrecords'





check_one_image_and_id(tfrecord_filename)
from quick_ml.visualize_and_check_data import check_batch_and_ids
tfrecord_filename = '/kaggle/working/unlabeled.tfrecords'

n_examples = 15

grid_rows = 3

grid_columns = 5

grid_size = (10,10)





check_batch_and_ids(tfrecord_filename, n_examples, grid_rows, grid_columns, grid_size)