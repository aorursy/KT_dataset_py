from kaggle_datasets import KaggleDatasets
GCS_PATH=KaggleDatasets().get_gcs_path("handwrittenbanglahocr")
print(GCS_PATH)
import os 


TFIDEN      =   'OcrDTF'# @param
DATA_DIM    =   128     # @param
NB_CHANNEL  =   3       # @param
BUFFER_SIZE =   3000    # @param
TRAIN_DATA  =   9600    # @param
EVAL_DATA   =   2400    # @param
TEST_DATA   =   3000    # @param
NB_CLASSES  =   50      # @param

GCS_PATH='{}/{}'.format(GCS_PATH,TFIDEN)
print(GCS_PATH)

train_path  = os.path.join(GCS_PATH, "Train")
val_path    = os.path.join(GCS_PATH, "Eval")
test_path   = os.path.join(GCS_PATH, "Test")
import tensorflow as tf 
import os 

print(tf.__version__)


def read_record(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, features)
    
    image = tf.image.decode_png(example["image"], channels=NB_CHANNEL)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [DATA_DIM,DATA_DIM,NB_CHANNEL])   
    label = tf.cast(example["label"], tf.int32)
    
    return image, label


AUTO = tf.data.experimental.AUTOTUNE

def prepare_dataset(flag, order = False):

    filenames = tf.io.gfile.glob(os.path.join(GCS_PATH, flag,"*.tfrecord"))
    print(flag,len(filenames))
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    
    # disregard the order of .tfrec files
    ignore_order = tf.data.Options()
    if order == False:
        ignore_order.experimental_deterministic = False
    else:
        ignore_order.experimental_deterministic = True
    dataset = dataset.with_options(ignore_order)
    
    dataset = dataset.map(read_record, num_parallel_calls=AUTO)
        
    return dataset

train_dataset = prepare_dataset("Train")
val_dataset   = prepare_dataset("Eval")
test_dataset  = prepare_dataset("Test")

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
for x,y in test_dataset.take(1):
    data=np.squeeze(x)
    plt.imshow(data)
    plt.show()
    print('Image Batch Shape:',x.shape)
    print('Target Batch Shape:',y.shape)
    print(y)
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

print("Number of TPU cores\t: ", tpu_strategy.num_replicas_in_sync)
print('Running on TPU\t\t: ', tpu.master())
# batch_size is scaled with the number of TPU cores
batch_size = 16 * tpu_strategy.num_replicas_in_sync
val_dataset = val_dataset.batch(batch_size).prefetch(AUTO)
test_dataset = test_dataset.batch(batch_size).prefetch(AUTO)
train_dataset = train_dataset.repeat().shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)
    


model_name='DenseNet121' # @param

with tpu_strategy.scope():
    base_model = tf.keras.applications.DenseNet121( include_top=False,
                                                    input_shape=(DATA_DIM,DATA_DIM,NB_CHANNEL),
                                                    weights='imagenet')
    base_model.trainable = True  
    
    model = Sequential(name="HOCR_MODEL")
    model.add(base_model)
    model.add(GlobalAveragePooling2D(name="GAP"))
    model.add(Dense(NB_CLASSES, activation="softmax", name="Probs"))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss="sparse_categorical_crossentropy", 
                 metrics=["sparse_categorical_accuracy"])

model.summary()
EPOCHS=250 # @param
TOTAL_DATA=TRAIN_DATA+EVAL_DATA
STEPS_PER_EPOCH = TOTAL_DATA//BATCH_SIZE
EVAL_STEPS      = EVAL_DATA//BATCH_SIZE
WEIGHT_PATH=os.path.join(os.getcwd(),f'{model_name}.h5') 

# reduces learning rate on plateau
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                  cooldown= 10,
                                                  patience=10,
                                                  verbose =1,
                                                  min_lr=0.1e-5)


# stop learining as metric on validatopn stop increasing
early_stopping = tf.keras.callbacks.EarlyStopping(patience=15, 
                                                  verbose=1, 
                                                  mode = 'auto') 

callbacks = [lr_reducer,early_stopping ]



history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    callbacks=[callbacks])
def plot_history(history):
    """
    Plots model training history 
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(history.epoch, history.history["accuracy"], label="Train IoU")
    ax_acc.plot(history.epoch, history.history["val_accuracy"], label="Validation IoU")
    ax_acc.legend()
    plt.show()

    # show history
plot_history(history)