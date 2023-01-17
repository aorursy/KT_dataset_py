!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.ex_tpu import *

step_1.check()
## let's load the data from the utility script

from petal_helper import *



import tensorflow as tf
# Lets learn the distribution startegy for the TPU's. 

# Each TPU has 8 cores (each core is like a GPU in itself)

# We need to tell tensorflow on how to make use of this TPU by a distribution strategy



# Detect TPU, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)

    

## Loading the data from the competition



# when using TPUs datasets are often serialized into TFRecords.

# This is a convenient format to feed to e ach of the TPU cores

# petal_helper utility script will load the TFRecords and create a data pipeline 

# to use with our model 





ds_train = get_training_dataset()

ds_valid = get_validation_dataset()

ds_test = get_test_dataset()



print("Training : ", ds_train)

print("Validation : ", ds_valid)

print("Testing : ", ds_test)



print("type : ", type(ds_test))

# These are tf.data.Dataset objects. You can think about the dataset in Tensorflow as a stream of data records

# We'll use Transfer learning where we use an already built pre-trained model

# and we can retrain a part of the models neural network to get a head-start on our new dataset



# The distribution strategy we created earlier contains a context manager, strategy.scope. 

# This context manager tells TensorFlow how to divide the work of training among the eight TPU cores. 

# When using TensorFlow with a TPU, it's important to define your model in a strategy.scope() context.





with strategy.scope():

    pretrained_model = tf.keras.applications.VGG16(

    weights = "imagenet",

    input_shape = [*IMAGE_SIZE, 3],

    include_top = False)

    

    

    pretrained_model.trainable = False

    

    model = tf.keras.Sequential([

    pretrained_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(len(CLASSES), activation='softmax')])

    

    model.compile(

    optimizer = 'adam',

    loss = 'sparse_categorical_crossentropy',

    metrics = ['sparse_categorical_accuracy'])

    



model.summary()

# Define the batch size. This will be 16 with TPU off and 128 (=16*8) with TPU on

BATCH_SIZE = 16*strategy.num_replicas_in_sync



# Defining the epochs

EPOCHS = 10

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



history = model.fit(

    ds_train,

    validation_data = ds_valid,

    epochs = EPOCHS,

    steps_per_epoch = STEPS_PER_EPOCH

    

)
display_training_curves(

    history.history['loss'],

    history.history['val_loss'],

    'loss',

    211,

)

display_training_curves(

    history.history['sparse_categorical_accuracy'],

    history.history['val_sparse_categorical_accuracy'],

    'accuracy',

    212,

)
test_ds = get_test_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)
## Let us generate submission.csv file.



print("generating submission.csv file ")





# Get image ids from test set and convert to unicode

test_ids_ds =  test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')





# Write submission.csv file

np.savetxt(

    'submission.csv',

    np.rec.fromarrays([test_ids, predictions]),

    fmt=['%s', '%d'],

    delimiter=',',

    header='id,label',

    comments='',

)
# Look at the first few predictions

!head submission.csv
