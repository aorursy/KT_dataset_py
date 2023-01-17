!pip install -q quick-ml
import tensorflow as tf

import quick_ml
from quick_ml.begin_tpu import define_tpu_strategy, get_training_dataset, get_validation_dataset, get_test_dataset
strategy, tpu = define_tpu_strategy()
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('catsdogstfrecords192x192')

print(GCS_DS_PATH)





train_tfrec_path = '/train.tfrecords'

val_tfrec_path = '/val.tfrecords'





BATCH_SIZE = 16*strategy.num_replicas_in_sync

EPOCHS = 6

NUM_TRAINING_IMAGES = 14961

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



print("STEPS PER EPOCH  => ", STEPS_PER_EPOCH)



dictionary_labeled = "{'image_raw' : tf.io.FixedLenFeature([], tf.string),'label' : tf.io.FixedLenFeature([], tf.int64)}"

IMAGE_SIZE = "192,192"



from quick_ml.begin_tpu import get_labeled_tfrecord_format

get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)
## LOADING TRAINING DATASET



train_dataset = get_training_dataset(GCS_DS_PATH, train_tfrec_path, BATCH_SIZE)
### LOADING VALIDATION DATASET



val_dataset = get_validation_dataset(GCS_DS_PATH, val_tfrec_path, BATCH_SIZE)
from quick_ml.load_models_quick import create_model
from quick_ml.callbacks import get_callbacks



callbacks = get_callbacks(lr_scheduler = 'rampup', early_stopping = None, reduce_lr_on_plateau = None)

print(callbacks)
with strategy.scope():

    model = create_model(1, model_name = 'EfficientNetB1', classification_model = 'default', freeze = True,

                         input_shape = [192,192,3], activation = 'sigmoid', weights = 'imagenet', optimizer = 'rmsprop', 

                        loss = 'binary_crossentropy', metrics = 'accuracy')
model.fit(train_dataset, 

         epochs = EPOCHS,

         steps_per_epoch = STEPS_PER_EPOCH, 

         validation_data = val_dataset, batch_size = BATCH_SIZE, callbacks = callbacks)
with strategy.scope():

    model2 = create_model(1, model_name = 'EfficientNetB0', classification_model = 'default', freeze = True,

                        input_shape = [192,192,3], activation = 'sigmoid', weights = 'imagenet', optimizer = 'rmsprop',

                        loss = 'binary_crossentropy', metrics = 'accuracy')
model2.fit(train_dataset, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH,

          validation_data = val_dataset, batch_size = BATCH_SIZE

          , callbacks = callbacks)
with strategy.scope():

    model3 = create_model(1, model_name = 'ResNet50V2', classification_model = 'default', freeze = True,

                         input_shape = [192,192,3], activation = 'sigmoid', weights = 'imagenet', 

                         optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = 'accuracy')
model3.fit(train_dataset, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_data = val_dataset,

          batch_size = BATCH_SIZE, callbacks = callbacks)
## obtain the GCS Path



from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('test-tfrecords-cats-dogs-192x192')

print('GCS_DS_PATH   -> ', GCS_DS_PATH)
# Define TFRecord Format for Unlabeled Data



from quick_ml.begin_tpu import get_unlabeled_tfrecord_format

dictionary_unlabeled = "{ 'image' : tf.io.FixedLenFeature([], tf.string), 'idnum' : tf.io.FixedLenFeature([], tf.string) }"

IMAGE_SIZE = "192,192"

get_unlabeled_tfrecord_format(dictionary_unlabeled, IMAGE_SIZE)


test_tfrec_path = '/test_cats_dogs_192x192.tfrecords'

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

test_dataset = get_test_dataset(GCS_DS_PATH, test_tfrec_path, BATCH_SIZE)
from quick_ml.predictions import ensemble_predictions
models = [model, model2, model3]

ensemble_predictions(models, test_dataset, ensemble_type = 'Model Averaging', classification_type = "binary")
import pandas as pd



df = pd.read_csv('./ensemble_model_averaging.csv')
df
ensemble_predictions(models, test_dataset, ensemble_type = 'Model Weighted', classification_type = 'binary', weights = [0.2,0.3,0.5])
df2 = pd.read_csv('/kaggle/working/ensemble_model_weighted.csv')

df2