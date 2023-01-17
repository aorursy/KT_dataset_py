!pip install -q quick-ml
import tensorflow as tf

import quick_ml



print("quick_ml version -> ", quick_ml.__version__)
from quick_ml.begin_tpu import define_tpu_strategy, get_training_dataset, get_validation_dataset, get_test_dataset
strategy, tpu = define_tpu_strategy()
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('catsdogstfrecords192x192')



print(GCS_DS_PATH)



train_tfrec_path = '/train.tfrecords'

val_tfrec_path = '/val.tfrecords'



BATCH_SIZE = 16 * strategy.num_replicas_in_sync



EPOCHS = 5

STEPS_PER_EPOCH = 14961 // BATCH_SIZE

print("STEPS PER EPOCH -> ", STEPS_PER_EPOCH)



dictionary_labeled = "{'image_raw' : tf.io.FixedLenFeature([], tf.string), 'label' : tf.io.FixedLenFeature([], tf.int64)}"

IMAGE_SIZE = "192,192"



from quick_ml.begin_tpu import get_labeled_tfrecord_format

get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)
from quick_ml.load_models_quick import create_model
with strategy.scope():

    model = create_model(1, model_name = 'EfficientNetB1', classification_model = 'default', freeze = False, 

                        input_shape =[192,192,3], activation = 'sigmoid', weights = 'imagenet', optimizer= 'rmsprop',

                        loss = 'binary_crossentropy', metrics = 'accuracy')

    
from quick_ml.augments import augment_and_train
from quick_ml.augments import define_augmentations
define_augmentations(flip_left_right = True, hue = 0.2, contrast = (0.1,0.4), brightness = 0.3)
from quick_ml.augments import define_callbacks



define_callbacks(lr_scheduler = 'rampup')
augment_and_train(model, GCS_DS_PATH, train_tfrec_path, val_tfrec_path, BATCH_SIZE, EPOCHS, STEPS_PER_EPOCH, plot = False)
with strategy.scope():

    model = create_model(1, model_name = 'EfficientNetB1', classification_model = 'default', freeze = False, 

                        input_shape =[192,192,3], activation = 'sigmoid', weights = 'imagenet', optimizer= 'rmsprop',

                        loss = 'binary_crossentropy', metrics = 'accuracy')
augment_and_train(model, GCS_DS_PATH, train_tfrec_path, val_tfrec_path, BATCH_SIZE, EPOCHS, STEPS_PER_EPOCH, plot = True)