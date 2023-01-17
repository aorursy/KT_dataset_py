import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(os.path.join(dirname))
import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.graph_objects as go

import cv2

import tensorflow as tf

from kaggle_datasets import KaggleDatasets

from functools import partial

import sklearn

from tqdm import tqdm_notebook as tqdm

import gc

%matplotlib inline
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()
print('Device:', tpu.master())

print('Number of replicas:', strategy.num_replicas_in_sync)

print("Version of Tensorflow used : ", tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE

GCS_PATH = KaggleDatasets().get_gcs_path()

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMAGE_SIZE = [1024, 1024]

SHAPE = [256, 256]
print("Batch Size = ", BATCH_SIZE)

print("GCS Path = ", GCS_PATH)
train = pd.DataFrame(pd.read_csv("../input/siim-isic-melanoma-classification/train.csv"))

train.head()
test = pd.DataFrame(pd.read_csv("../input/siim-isic-melanoma-classification/test.csv"))

test.head()
train.info()
test.info()
train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
image_names = train["image_name"].values + ".jpg"

random_images = [np.random.choice(image_names) for i in range(4)] # Generates a random sample from a given 1-D array

random_images 
sample_images = []
plt.figure(figsize = (12, 8))

for i in range(4) : 

    plt.subplot(2, 2, i + 1) 

    image = cv2.imread(os.path.join(train_dir, random_images[i]))

    # cv2 reads images in BGR format. Hence we convert it to RGB

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sample_images.append(image)

    plt.imshow(image, cmap = "gray")

    plt.grid(True)

# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
def non_local_means_denoising(image) : 

    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return denoised_image
sample_image = cv2.imread(os.path.join(train_dir, random_images[0]))

# cv2 reads images in BGR format. Hence we convert it to RGB

sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

denoised_image = non_local_means_denoising(sample_image)





plt.figure(figsize = (12, 8))

plt.subplot(1,2,1)

plt.imshow(sample_image, cmap = "gray")

plt.grid(False)

plt.title("Normal Image")



plt.subplot(1,2,2)  

plt.imshow(denoised_image, cmap = "gray")

plt.grid(False)

plt.title("Denoised image")    

# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout() 
def histogram_equalization(image) : 

    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)

    y_channel = image_ycrcb[:,:,0] # apply local histogram processing on this channel

    cr_channel = image_ycrcb[:,:,1]

    cb_channel = image_ycrcb[:,:,2]

    

    # Local histogram equalization

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))

    equalized = clahe.apply(y_channel)

    equalized_image = cv2.merge([equalized, cr_channel, cb_channel])

    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_YCR_CB2RGB)

    return equalized_image
equalized_image = histogram_equalization(denoised_image)
plt.figure(figsize = (12, 8))

plt.subplot(1,3,1)

plt.imshow(sample_image, cmap = "gray")

plt.grid(False)

plt.title("Normal Image", fontsize = 14)



plt.subplot(1,3,2)  

plt.imshow(denoised_image, cmap = "gray")

plt.grid(False)

plt.title("denoised image after histogram processing", fontsize = 14)



plt.subplot(1,3,3)  

plt.imshow(equalized_image, cmap = "gray")

plt.grid(False)

plt.title("Histogram equalized image", fontsize = 14)

# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
def segmentation(image, k, attempts) : 

    vectorized = np.float32(image.reshape((-1, 3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    res , label , center = cv2.kmeans(vectorized, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]

    segmented_image = res.reshape((image.shape))

    return segmented_image
plt.figure(figsize = (12, 8))

plt.subplot(1,1,1)

plt.imshow(denoised_image, cmap = "gray")

plt.grid(False)

plt.title("de Noised Image")
plt.figure(figsize = (12, 8))

segmented_image = segmentation(denoised_image, 3, 10) # k = 3, attempt = 10

plt.subplot(1,3,1)

plt.imshow(segmented_image, cmap = "gray")

plt.grid(False)

plt.title("Segmented Image with k = 3")



segmented_image = segmentation(denoised_image, 4, 10) # k = 4, attempt = 10

plt.subplot(1,3,2)

plt.imshow(segmented_image, cmap = "gray")

plt.grid(False)

plt.title("Segmented Image with k = 4")



segmented_image = segmentation(denoised_image, 5, 10) # k = 5, attempt = 10

plt.subplot(1,3,3)

plt.imshow(segmented_image, cmap = "gray")

plt.grid(False)

plt.title("Segmented Image with k = 5")
from sklearn.model_selection import train_test_split 

training_files, validation_files = train_test_split(tf.io.gfile.glob(GCS_PATH + "/tfrecords/train*.tfrec"),

                                                   test_size = 0.1, random_state = 42)



testing_files = tf.io.gfile.glob(GCS_PATH + "/tfrecords/test*.tfrec")



print("Number of training files = ", len(training_files))

print("Number of validation files = ", len(validation_files))

print("Number of test files = ", len(testing_files))
def decode_image(image) : 

    image = tf.image.decode_jpeg(image, channels = 3)

    image = tf.cast(image, tf.float32)

    image = image / 255.0

    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

    return image
sample_images[0].shape
training_files
sample_picked = training_files[0]

sample_picked
file = tf.data.TFRecordDataset(sample_picked)

file
feature_description = {"image" : tf.io.FixedLenFeature([], tf.string), 

                      "target" : tf.io.FixedLenFeature([], tf.int64)}
def parse_function(example) : 

    # The example supplied is parsed based on the feature_description above.

    return tf.io.parse_single_example(example, feature_description)
parsed_dataset = file.map(parse_function)

parsed_dataset
def read_tfrecord(example, labeled) : 

    if labeled == True : 

        tfrecord_format = {"image" : tf.io.FixedLenFeature([], tf.string),

                           "target" : tf.io.FixedLenFeature([], tf.int64)}

    else:

        tfrecord_format = {"image" : tf.io.FixedLenFeature([], tf.string),

                          "image_name" : tf.io.FixedLenFeature([], tf.string)}

    

    example = tf.io.parse_single_example(example, tfrecord_format)

    image = decode_image(example["image"])

    if labeled == True : 

        label = tf.cast(example["target"], tf.int32)

        return image, label

    else:

        image_name = example["image_name"]

        return image, image_name     
def load_dataset(filenames, labeled, ordered):

    ignore_order = tf.data.Options()

    if ordered == False: # dataset is unordered, so we ignore the order to load data quickly.

        ignore_order.experimental_deterministic = False # This disables the order and enhances the speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) 

    dataset = dataset.with_options(ignore_order) 

    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)

    return dataset
def image_augmentation(image, label) :     

    image = tf.image.resize(image, SHAPE)

    image = tf.image.random_flip_left_right(image)

    return image, label
def get_training_dataset() : 

    dataset = load_dataset(training_files, labeled = True, ordered = False)

    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)

    dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE) 

    return dataset
def get_validation_dataset() : 

    dataset = load_dataset(validation_files, labeled = True, ordered = False)

    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTOTUNE) 

    return dataset
def get_test_dataset() : 

    dataset = load_dataset(testing_files, labeled = False, ordered = True)

    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTOTUNE) 

    return dataset
training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()
def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



num_training_images = count_data_items(training_files)

num_validation_images = count_data_items(validation_files)

num_testing_images = count_data_items(testing_files)



STEPS_PER_EPOCH_TRAIN = num_training_images // BATCH_SIZE

STEPS_PER_EPOCH_VAL = num_validation_images // BATCH_SIZE



print("Number of Training Images = ", num_training_images)

print("Number of Validation Images = ", num_validation_images)

print("Number of Testing Images = ", num_testing_images)

print("\n")

print("Numer of steps per epoch in Train = ", STEPS_PER_EPOCH_TRAIN)

print("Numer of steps per epoch in Validation = ", STEPS_PER_EPOCH_VAL)
image_batch, label_batch = next(iter(training_dataset))
def show_batch(image_batch, label_batch) :

    plt.figure(figsize = (20, 20))

    for n in range(8) : 

        ax = plt.subplot(2,4,n+1)

        plt.imshow(image_batch[n])

        if label_batch[n] == 0 : 

            plt.title("BENIGN")

        else:

            plt.title("MALIGNANT")

    plt.grid(False)

    plt.tight_layout()       
show_batch(image_batch.numpy(), label_batch.numpy())
del image_batch

del label_batch

gc.collect()
malignant = len(train[train["target"] == 1])

benign = len(train[train["target"] == 0 ])

total = len(train) 



print("Malignant Cases in Train Data = ", malignant)

print("Benign Cases In Train Dataset = ",benign)

print("Total Cases In Train Dataset = ",total)

print("Ratio of Malignant to Benign = ",malignant/benign)
weight_malignant = (total/malignant)/2.0

weight_benign = (total/benign)/2.0



class_weight = {0 : weight_benign , 1 : weight_malignant}



print("Weight for benign cases = ", class_weight[0])

print("Weight for malignant cases = ", class_weight[1])
callback_early_stopping = tf.keras.callbacks.EarlyStopping(patience = 15, verbose = 0, restore_best_weights = True)



callbacks_lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_auc", factor = 0.1, patience = 10, 

                                                          verbose = 0, min_lr = 1e-6)



callback_checkpoint = tf.keras.callbacks.ModelCheckpoint("melanoma_weights.h5",

                                                         save_weights_only=True, monitor='val_auc',

                                                         mode='max', save_best_only = True)
with strategy.scope() : 

    bias = np.log(malignant/benign)

    bias = tf.keras.initializers.Constant(bias)

    base_model = tf.keras.applications.MobileNetV2(input_shape = (SHAPE[0], SHAPE[1], 3), include_top = False,

                                               weights = "imagenet")

    base_model.trainable = False

    model = tf.keras.Sequential([base_model,

                                 tf.keras.layers.GlobalAveragePooling2D(),

                                 tf.keras.layers.Dense(20, activation = "relu"),

                                 tf.keras.layers.Dropout(0.4),

                                 tf.keras.layers.Dense(10, activation = "relu"),

                                 tf.keras.layers.Dropout(0.3),

                                 tf.keras.layers.Dense(1, activation = "sigmoid", bias_initializer = bias)                                     

                                ])

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-2), loss = "binary_crossentropy", metrics = [tf.keras.metrics.AUC(name = 'auc')])

    model.summary()

    

    EPOCHS = 500

    history = model.fit(training_dataset, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH_TRAIN,

                       validation_data = validation_dataset, validation_steps = STEPS_PER_EPOCH_VAL,

                       callbacks = [callback_early_stopping, callbacks_lr_reduce, callback_checkpoint],

                       class_weight = class_weight)
n_epochs_it_ran_for = len(history.history['loss'])

n_epochs_it_ran_for
X = np.arange(0,n_epochs_it_ran_for,1)

plt.figure(1, figsize = (20, 12))

plt.subplot(1,2,1)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.plot(X, history.history["loss"], label = "Training Loss")

plt.plot(X, history.history["val_loss"], label = "Validation Loss")

plt.grid(True)

plt.legend()



plt.subplot(1,2,2)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.plot(X, history.history["auc"], label = "Training Accuracy")

plt.plot(X, history.history["val_auc"], label = "Validation Accuracy")

plt.grid(True)

plt.legend()
testing_dataset = get_test_dataset()

testing_dataset_images = testing_dataset.map(lambda image, image_name : image)

testing_image_names = testing_dataset.map(lambda image, image_name : image_name)
resulting_probabilities = model.predict(testing_dataset_images, verbose = 1)
len(resulting_probabilities)
sample_submission_file = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")

sample_submission_file.head()
del sample_submission_file["target"]

sample_submission_file.head()
testing_image_names
testing_image_names = np.concatenate([x for x in testing_image_names], axis=0)

testing_image_names = np.array(testing_image_names)
decoded_test_names = []

for names in testing_image_names : 

    names = names.decode('utf-8')

    decoded_test_names.append(names)

decoded_test_names = np.array(decoded_test_names)

del testing_image_names
len(decoded_test_names), type(decoded_test_names), decoded_test_names.shape
decoded_test_names
testing_image_names = pd.DataFrame(decoded_test_names, columns=["image_name"])

testing_image_names.head()
pred_dataframe = pd.DataFrame({"image_name" : decoded_test_names, 

                               "target" : np.concatenate(resulting_probabilities)})

pred_dataframe
sample_submission_file = sample_submission_file.merge(pred_dataframe, on = "image_name")

sample_submission_file.to_csv("submission.csv", index = False)

sample_submission_file.head()