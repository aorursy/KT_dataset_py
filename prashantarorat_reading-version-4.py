from kaggle_datasets import KaggleDatasets
import tensorflow as tf
GCS_PATH = KaggleDatasets().get_gcs_path()
GCS_PATH

train_filenames = tf.io.gfile.glob(GCS_PATH+"/train*.tfrec")
train_filenames
