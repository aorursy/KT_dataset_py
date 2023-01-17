!pip install -q quick-ml
import tensorflow as tf

import quick_ml
from quick_ml.tfrecords_maker import create_split_tfrecords_data



!wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip -O catsdogs.zip

!unzip -q catsdogs.zip

!rm catsdogs.zip
data_dir = '/kaggle/working/PetImages'

outfile1name = 'train.tfrecords'

output1folder = 'train'

outfile2name = 'val.tfrecords'

output2folder = 'val'

split_size_ratio = 0.8



create_split_tfrecords_data(data_dir, outfile1name,output1folder,  outfile2name,output2folder,  split_size_ratio,num_parts1 = 10, num_parts2 = 5,  IMAGE_SIZE = (192,192))
from quick_ml.visualize_and_check_data import check_batch_and_labels
dictionary_labeled = "{'image' : tf.io.FixedLenFeature([], tf.string),'label' : tf.io.FixedLenFeature([], tf.int64)}"

IMAGE_SIZE = "192,192"



from quick_ml.begin_tpu import get_labeled_tfrecord_format



get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)
check_batch_and_labels('/kaggle/working/train/train_part_3.tfrecords', 9, 3, 3, grid_size=(15, 15))
check_batch_and_labels('/kaggle/working/val/val_part_3.tfrecords', 9, 3, 3, grid_size=(15, 15))