!pip install -q quick-ml
import tensorflow as tf

import quick_ml
from quick_ml.k_fold_training import train_k_fold_pred
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('cats-dogs-192x192-tfrecords-part-wise')

print(GCS_DS_PATH)
train_tfrec_path = '/Train/*.tfrecords'

val_tfrec_path = '/Val/*.tfrecords'

#TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + train_tfrec_path) + tf.io.gfile.glob(GCS_DS_PATH + val_tfrec_path)
#TRAINING_FILENAMES
import pandas as pd
from quick_ml.begin_tpu import get_labeled_tfrecord_format



dictionary_labeled =  "{'image' : tf.io.FixedLenFeature([], tf.string),'label' : tf.io.FixedLenFeature([], tf.int64)}"

IMAGE_SIZE = "192,192"



get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)
from quick_ml.begin_tpu import define_tpu_strategy



strategy, tpu = define_tpu_strategy()
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH_labeled = KaggleDatasets().get_gcs_path('cats-dogs-192x192-tfrecords-part-wise')

GCS_DS_PATH_labeled
train_tfrecs_path = '/Train/train*.tfrecords'

val_tfrecs_path = '/Val/val*.tfrecords'
dictionary_labeled =  "{'image' : tf.io.FixedLenFeature([], tf.string),'label' : tf.io.FixedLenFeature([], tf.int64)}"

IMAGE_SIZE = "192,192"

from quick_ml.begin_tpu import get_labeled_tfrecord_format

get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH_unlabeled = KaggleDatasets().get_gcs_path('test-tfrecords-cats-dogs-192x192')

GCS_DS_PATH_unlabeled
test_tfrec_path = '/test*.tfrecords'
from quick_ml.begin_tpu import get_unlabeled_tfrecord_format

dictionary_unlabeled = "{ 'image' : tf.io.FixedLenFeature([], tf.string), 'idnum' : tf.io.FixedLenFeature([], tf.string) }"

IMAGE_SIZE = "192,192"

get_unlabeled_tfrecord_format(dictionary_unlabeled, IMAGE_SIZE)
help(train_k_fold_pred)
k = 5

n_class = 1

model_name = 'EfficientNetB1'



test_tfrecs_path = '/test*.tfrecords'

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

EPOCHS = 5

input_shape = [192,192,3]

activation = 'sigmoid'

optimizer = 'rmsprop'

loss = 'binary_crossentropy'

metrics = 'accuracy'



effnetb1 = train_k_fold_pred(k, tpu,n_class,  model_name, train_tfrecs_path, val_tfrecs_path, test_tfrecs_path, 

                            GCS_DS_PATH_labeled, GCS_DS_PATH_unlabeled, BATCH_SIZE, EPOCHS, input_shape = input_shape, 

                            activation = activation, optimizer = optimizer, loss = loss, metrics = metrics)
df = effnetb1.train_k_fold()
df
df.to_csv("training_report.csv", index = False)
effnetb1.obtain_predictions()
preds = pd.read_csv('predictions.csv')

preds