!pip install -q quick_ml
import tensorflow as tf
import quick_ml
from quick_ml.begin_tpu import define_tpu_strategy
strategy, tpu = define_tpu_strategy()
from quick_ml.callbacks import get_callbacks
callbacks = get_callbacks(lr_scheduler = 'rampup', early_stopping = 'default', reduce_lr_on_plateau = 'default')

callbacks
from quick_ml.load_models_quick import create_model
with strategy.scope():

    model = create_model(4, input_shape = [192,192,3])

    

    callbacks = get_callbacks(lr_scheduler = 'rampup', early_stopping = 'default', reduce_lr_on_plateau = 'default')

    
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('4classtfrecords-latest')
EPOCHS = 8

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

STEPS_PER_EPOCH = 3496 // BATCH_SIZE
from quick_ml.begin_tpu import get_labeled_tfrecord_format
dictionary_labeled = "{'image' : tf.io.FixedLenFeature([], tf.string),'label' : tf.io.FixedLenFeature([], tf.int64)}"

IMAGE_SIZE = "192,192"



get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)

from quick_ml.begin_tpu import get_training_dataset, get_validation_dataset
train_tfrec_path = '/train.tfrecords'



traindata = get_training_dataset(GCS_DS_PATH, train_tfrec_path, BATCH_SIZE)
val_tfrec_path = '/val.tfrecords'



val_data = get_validation_dataset(GCS_DS_PATH, val_tfrec_path, BATCH_SIZE)
model.fit(traindata, validation_data = val_data, callbacks = callbacks,steps_per_epoch = 28, epochs = 10)
from quick_ml.training_predictions import get_models_training_report
models = ['VGG16', 'DenseNet121', 'EfficientNetB6']
df = get_models_training_report(models,tpu, 4, traindata, STEPS_PER_EPOCH, EPOCHS, BATCH_SIZE, val_data,  classification_model = 'default', freeze = False, input_shape = [192,192,3], activation = 'softmax', weights = "imagenet", optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = "sparse_categorical_accuracy", callbacks = callbacks, plot = False)
df
df.to_csv("outputs_callbacks.csv", index = False)