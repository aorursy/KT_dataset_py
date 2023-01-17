# Installing EfficientNet module 
!pip install efficientnet
# Computation Specific
import re
import os
import numpy as np
import pandas as pd
from functools import partial

# Machine Learning Specific
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger


# Augmentation and Visualization Specific
import imgaug
import matplotlib.pyplot as plt

# Pretrained Model
import efficientnet.tfkeras as efn


TPU_used = True
if TPU_used:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    print('Number of replicas:', strategy.num_replicas_in_sync)
    
print(tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = KaggleDatasets().get_gcs_path()
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(
    tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec'),
    test_size=0.2, random_state=42
)
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')
print('Train TFRecord Files:', len(TRAINING_FILENAMES))
print('Validation TFRecord Files:', len(VALID_FILENAMES))
print('Test TFRecord Files:', len(TEST_FILENAMES))
def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image
def read_tfrecord(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum
def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=AUTOTUNE) 
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled),num_parallel_calls=AUTOTUNE)
    if not ordered:
        ignore_order.experimental_deterministic = False 
        dataset.with_options(ignore_order)
    return dataset
def augmentation_pipeline(image,label):
    image = tf.image.convert_image_dtype(image,tf.float32)
    image  =tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image,max_delta=0.5)    
    return image,label

#image = tf.image.random_brightness(image,max_delta=0.3)
#image = tf.image.random_flip_left_right(image)
#image  =tf.image.random_flip_up_down(image)
#image = tf.image.random_saturation(image,3,8)
class Dataset():
    
    def __init__(self,batch_size):
        self.batch_size = batch_size
    
    def get_dataset(self,filenames,labeled,ordered,mode="train"):
        
        assert(mode in ["train","valid","test"])    # Checking if the mode is one of the correct ones.
        
        # Prepairing dataset according to the provided mode.
        
        dataset = load_dataset(filenames, labeled=labeled,ordered=ordered)
        if mode=="train":
            dataset = dataset.map(augmentation_pipeline)
            dataset = dataset.repeat()
            dataset = dataset.shuffle(2048)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
                                                    
        if mode=="valid":
            dataset = dataset.cache()
            
        return dataset
    
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALID_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print(
    'Dataset: {} training images, {} validation images, {} unlabeled test images and the steps per epoch is {}'.format(
        NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES, STEPS_PER_EPOCH
    )
)
# Note that learning rate first increase till 10 epochs and then reduces as the epoch increases.
def build_lrfn(lr_start=0.00001, lr_max=0.000075, lr_min=0.0000001, lr_rampup_epochs=5, lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    return lrfn
def Callbacks():
    cb = []
    
    # Drop based LR Scheduling.
    reducelr = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=5, min_lr = 1e-06, factor=0.2)
    log = CSVLogger("Melanoma_classification.csv")
    cb.append(reducelr)
    cb.append(log)
    return cb
def make_model(output_bias = None, metrics = None):
    
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    base_model = efn.EfficientNetB7(input_shape=(*IMAGE_SIZE, 3),
                                             include_top=False,
                                             weights="imagenet")
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16,activation="relu"),
        tf.keras.layers.Dense(8,activation="relu"),
        tf.keras.layers.Dense(1, activation='sigmoid',
                              bias_initializer=output_bias
                             )
    ])
    
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                                                  0.001,
                                                  decay_steps=(NUM_TRAINING_IMAGES // BATCH_SIZE)*100,
                                                  decay_rate=1,
                                                  staircase=False
                                                )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='binary_crossentropy',
                  metrics=[metrics])
    
    return model
dataset = Dataset(BATCH_SIZE)
train_dataset = dataset.get_dataset(TRAINING_FILENAMES,labeled = True,ordered = False,mode="train")
valid_dataset = dataset.get_dataset(VALID_FILENAMES,labeled = True,ordered = False,mode="valid")
train_csv = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_csv = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

total_img = train_csv['target'].size

malignant = np.count_nonzero(train_csv['target'])
benign = total_img - malignant

print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total_img, malignant, 100 * malignant / total_img))
initial_bias = np.log([malignant/benign])
initial_bias
weight_for_0 = (1 / benign)*(total_img)/2.0 
weight_for_1 = (1 / malignant)*(total_img)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
lrfn = build_lrfn()
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALID_STEPS = NUM_VALIDATION_IMAGES // BATCH_SIZE
with strategy.scope():
    model = make_model(metrics=tf.keras.metrics.AUC(name='auc'),output_bias=initial_bias)
callback = Callbacks()

history = model.fit(
    train_dataset, epochs=20,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset,
    validation_steps=VALID_STEPS,
    callbacks= [tf.keras.callbacks.CSVLogger("Melanoma_classification.csv")],
    class_weight=class_weight,
    verbose=1
)
# tf.keras.callbacks.LearningRateScheduler(lrfn)
# tf.keras.callbacks.CSVLogger("Melanoma_classification_new.csv")
## Visualizing Model Performance
def visualize(epochs,x,y,label=""):
    plt.plot(epochs,x,marker="o",c="red",label=f"Training {label}")
    plt.plot(epochs,y,marker="x",c="green",label=f"Validation {label}")
    plt.legend()
    plt.grid(False)

visualize_df = pd.read_csv("/kaggle/working/Melanoma_classification.csv")
epochs       = visualize_df["epoch"]
auc          = visualize_df["auc"]
val_auc      = visualize_df["val_auc"]
loss         = visualize_df["loss"]
val_loss     = visualize_df["val_loss"]

fig = plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
visualize(epochs,auc,val_auc,label="AUC")
plt.subplot(1,2,2)
visualize(epochs,loss,val_loss,label="LOSS")
plt.tight_layout()
plt.show()
# Getting test Dataset.
# Note that here test dataset is ordered. 
test_ds = dataset.get_dataset(TEST_FILENAMES,labeled=False,ordered=True,mode="test")
%%time
def submit(filename="submission.csv"):
    print('Generating submission.csv file...')
    print('Computing predictions...')
    test_images_ds = test_ds.map(lambda image, idnum: image)
    probabilities = model.predict(test_images_ds)
    repr("Prediction done")
    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
    pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
    sub = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv")
    del sub['target']
    sub = sub.merge(pred_df, on='image_name')
    sub.to_csv(open(filename,"w"), index=False)
    print("Submitted Successfully.....")
submit()
