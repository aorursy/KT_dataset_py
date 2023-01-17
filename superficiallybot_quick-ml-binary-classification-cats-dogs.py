!pip install -q quick-ml
# maintain the order of the imports



import tensorflow as tf

import quick_ml
from quick_ml.begin_tpu import define_tpu_strategy, get_training_dataset, get_validation_dataset, get_test_dataset
strategy, tpu = define_tpu_strategy()
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('catsdogstfrecords192x192')

print(GCS_DS_PATH)



train_tfrec_path = '/train.tfrecords'

val_tfrec_path = '/val.tfrecords'



BATCH_SIZE = 16 * strategy.num_replicas_in_sync



EPOCHS = 6

STEPS_PER_EPOCH = 14961 // BATCH_SIZE

print("STEPS PER EPOCH -> ", STEPS_PER_EPOCH)



dictionary_labeled = "{'image_raw' : tf.io.FixedLenFeature([], tf.string), 'label' : tf.io.FixedLenFeature([], tf.int64)}"

IMAGE_SIZE = "192,192"



from quick_ml.begin_tpu import get_labeled_tfrecord_format

get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)

train_dataset = get_training_dataset(GCS_DS_PATH, train_tfrec_path, BATCH_SIZE)
val_dataset = get_validation_dataset(GCS_DS_PATH, val_tfrec_path, BATCH_SIZE)
from quick_ml.load_models_quick import create_model
with strategy.scope():

    model = create_model(1, model_name = 'EfficientNetB1', classification_model = 'default', freeze = False, input_shape = [192,192,3], activation = 'sigmoid', weights = 'imagenet', optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = 'accuracy')
model.fit(train_dataset, 

         epochs = EPOCHS, 

         steps_per_epoch =STEPS_PER_EPOCH,

         validation_data = val_dataset, batch_size = BATCH_SIZE)
from quick_ml.training_predictions import get_models_training_report
models = ['VGG16', 'InceptionV3', 'DenseNet201', 'EfficientNetB1']
df1 = get_models_training_report(models, tpu, 1, train_dataset, STEPS_PER_EPOCH, EPOCHS, BATCH_SIZE, val_dataset, classification_model = 'default', freeze = False, input_shape = [192,192,3], activation = 'sigmoid', weights = 'imagenet', optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = 'accuracy')
df1
df1.to_csv('output.csv', index = False)