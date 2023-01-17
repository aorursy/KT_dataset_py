!pip install quick_ml
import tensorflow as tf
import quick_ml
from quick_ml.begin_tpu import define_tpu_strategy
strategy, tpu = define_tpu_strategy()
# If TPU is activated and everything went good, this line of code must print 8 as output.



strategy.num_replicas_in_sync
from quick_ml.load_models_quick import create_model
with strategy.scope():

    model = create_model(4, model_name = 'VGG19', classification_model = 'default', freeze = False, input_shape = [192, 192,3], activation  = 'softmax', weights= "imagenet", optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'sparse_categorical_accuracy')
model.summary()
# SETUP the Hyper Parameters



from quick_ml.begin_tpu import get_training_dataset

from quick_ml.begin_tpu import get_validation_dataset



EPOCHS = 6 # Small number of epochs to quickly finish off training. You can set number of epochs as per your requirement.

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

STEPS_PER_EPOCH = 3496 // BATCH_SIZE

print(f"Steps per epoch -> {STEPS_PER_EPOCH}")



from kaggle_datasets import KaggleDatasets



GCS_DS_PATH = KaggleDatasets().get_gcs_path('4classtfrecords-latest')



print(f'GCS_DS_PATH -> {GCS_DS_PATH}')



dictionary_labeled = "{ 'image' : tf.io.FixedLenFeature([], tf.string), 'label' : tf.io.FixedLenFeature([], tf.int64) }"

IMAGE_SIZE = "192,192"



from quick_ml.begin_tpu import get_labeled_tfrecord_format



get_labeled_tfrecord_format(dictionary_labeled, IMAGE_SIZE)



"""

for unlabeled tfrecords data, do the following...



-------------------------------------------------------------------------------------------

dictionary_unlabeled = "COPY THE UNLABELED TFRECORD FORMAT FOR YOUR DATASET (UNLABELED) AND PUT IT IN A STRING"

IMAGE_SIZE = "dim1,dim2"



from quick_ml.begin_tpu import get_unlabeled_tfrecord_format

get_unlabeled_tfrecord_format(dictionary_unlabeled, IMAGE_SIZE)

-------------------------------------------------------------------------------------------------





NOTE :- For loading of the tfrecord datasets (labeled or unlabeled), any deviations for the variable names or the value assignment,

the functions will pop up error(s).



"""

train_tfrec_path = '/train.tfrecords'   # Be careful with this

val_tfrec_path = '/val.tfrecords'       # as well as this, full file path doesn't work.



traindata = get_training_dataset(GCS_DS_PATH, train_tfrec_path , 128)

val_data = get_validation_dataset(GCS_DS_PATH, val_tfrec_path , 128)
history = model.fit( traindata, 

    steps_per_epoch = STEPS_PER_EPOCH, 

    epochs = EPOCHS,

          batch_size = BATCH_SIZE,

    validation_data = val_data

)
from quick_ml.training_predictions import get_models_training_report
# define the models list



models = ['VGG16', 'VGG19',  

    'Xception',

    'DenseNet121', 'DenseNet169', 'DenseNet201', 

    'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 

    'MobileNet', 'MobileNetV2',

    'InceptionV3', 'InceptionResNetV2', 

    'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']
print(f'Total number of models -> {len(models)}')
n_class = 4

df = get_models_training_report(models,tpu, n_class, traindata, STEPS_PER_EPOCH, EPOCHS, BATCH_SIZE, val_data,  classification_model = 'default', freeze = False, input_shape = [192,192,3], activation = 'softmax', weights = "imagenet", optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = "sparse_categorical_accuracy", plot = False)
df
df.to_csv('outputs.csv', index = False)