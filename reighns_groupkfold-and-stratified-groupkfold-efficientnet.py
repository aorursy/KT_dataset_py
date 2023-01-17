!pip install -q efficientnet
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

import os 
import re
import math
from matplotlib import pyplot as plt
from math import ceil
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn import model_selection

import tensorflow as tf
import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

from kaggle_datasets import KaggleDatasets

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Defining data path
train_images_dir = '../input/siim-isic-melanoma-classification/train/'
test_images_dir = '../input/siim-isic-melanoma-classification/test/'
train_csv = '../input/siim-isic-melanoma-classification/train.csv'
test_csv  = '../input/siim-isic-melanoma-classification/test.csv'
sample_submission = '../input/siim-isic-melanoma-classification/sample_submission.csv'

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

train_df.head()

print("The unique number of patiend_ids are {}".format(train_df['patient_id'].nunique()))
group_by_patient_id = train_df.groupby(['patient_id', 'image_name']) 
group_by_patient_id.first()
groups_by_patient_id_list = train_df['patient_id'].copy().tolist()
# the below code should work better in fact
# groups_by_patient_id_list = np.array(train_df['patient_id'].values)

y_labels = train_df["target"].values
# x_train = train_df[["image_name","patient_id","sex","age_approx","anatom_site_general_challenge"]]
# y_train = train_df[["target"]]


n_splits = 5
gkf = GroupKFold(n_splits = 5)

result = []   
for train_idx, val_idx in gkf.split(train_df, y_labels, groups = groups_by_patient_id_list):
    train_fold = train_df.iloc[train_idx]
    val_fold = train_df.iloc[val_idx]
    result.append((train_fold, val_fold))
    
train_fold_1, val_fold_1 = result[0][0],result[0][1]
train_fold_2, val_fold_2 = result[1][0],result[1][1]
train_fold_3, val_fold_3 = result[2][0],result[2][1]
train_fold_4, val_fold_4 = result[3][0],result[3][1]
train_fold_5, val_fold_5 = result[4][0],result[4][1]



# just to check if it works as intended
sample = train_fold_1.groupby("patient_id")
sample.get_group("IP_0147446")
sample.get_group("IP_0147446").count()
# sample2 = val_fold_1.groupby("patient_id")
# sample2.get_group("IP_0063782")
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

# Configuration
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
image_size = 256
EPOCHS = 3
def format_path_train(img_name):
    return GCS_PATH + '/jpeg/train/' + img_name + '.jpg'

def format_path_test(img_name):
    return GCS_PATH + '/jpeg/test/' + img_name + '.jpg'
train_paths_fold_1 = train_fold_1.image_name.apply(format_path_train).values
val_paths_fold_1 = val_fold_1.image_name.apply(format_path_train).values

train_labels_fold_1 = train_fold_1.target.values
val_labels_fold_1 = val_fold_1.target.values

test_paths = test_df.image_name.apply(format_path_test).values
def decode_image(filename, label=None, image_size=(image_size, image_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, size = image_size)
    
    if label is None:
        return image
    else:
        return image, label

# def data_augment(image, label=None):
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)
    
#     if label is None:
#         return image
#     else:
#         return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
#     image = tf.image.random_saturation(image, lower = 1, upper = 3)
#     image = tf.image.adjust_brightness(image, delta = 0.3)
    image = tf.image.random_contrast(image, lower = 1, upper = 2)
    if label is None:
        return image
    else:
        return image, label
train_paths_fold_1
sample_filename = 'gs://kds-fbc9e41d779a5db41c5e7caf080daf3b4d85265f605fdbc4ce15cbfa/jpeg/train/ISIC_2637011.jpg' 
sample_label = 0
image_size = 256

# 1. tf.io_read_file takes in a Tensor of type string and outputs a ensor of type string. 
#    Reads and outputs the entire contents of the input filename. 
bits = tf.io.read_file(sample_filename)

# 2. Decode a JPEG-encoded image to a uint8 tensor. You can also use tf.io.decode_jpeg but according to 
#    tensorflow's website, it might be cleaner to use tf.image.decode_jpeg
image = tf.image.decode_jpeg(bits, channels=3)

image.shape  # outputs TensorShape([4000, 6000, 3])

# 3. image = tf.cast(image, tf.float32) / 255.0 is easy to understand, it takes in 
#    an image, and cast the image into the data type you want. Here we also normalized by dividing by 255.

image = tf.cast(image, tf.float32) / 255.0


# 4. image = tf.image.resize(image, image_size) is also easy to understand. We merely resize this image to the image_size we wish for.
#    take note in our function defined above, the argument image_size is a tuple already. So we must pass in a tuple of our desired image_size.
image = tf.image.resize(image, size = (image_size, image_size))

image.shape
# example from tensorflow's website
sample_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for element in sample_dataset:
    print(element)
dataset = tf.data.Dataset.from_tensor_slices((train_paths_fold_1, train_labels_fold_1))
for data in dataset:
    print(len(data))
    print(data[0])
    print(data[1])   
    break
dataset = tf.data.Dataset.from_tensor_slices((train_paths_fold_1, train_labels_fold_1)).map(decode_image, num_parallel_calls=AUTO)
for data in dataset:
    print(len(data))
    print(data[0])
    print(data[1])
    break
sample_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
sample_dataset = sample_dataset.repeat(3)
list(sample_dataset)
# dataset = tf.data.Dataset.from_tensor_slices((train_paths_fold_1, train_labels_fold_1)).map(decode_image, num_parallel_calls=AUTO).repeat()
# # for data in dataset:
# #     print(len(data))
# #     print(data[0])
# #     print(data[1])
# #     break
# dataset = tf.data.Dataset.from_tensor_slices((train_paths_fold_1, train_labels_fold_1)).map(decode_image, num_parallel_calls=AUTO).repeat().batch(32)
# # here it returns 32 images and its labels, because we specified our batch size to be 32! 
# for data in dataset:
#     print(data)
#     break
image_folder_path = '../input/siim-isic-melanoma-classification/jpeg/train/'
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0074542.jpg"))[:,:,::-1]
plt.imshow(chosen_image)
horizontal_flipped = tf.image.flip_left_right(chosen_image)
vertically_flipped = tf.image.flip_up_down(chosen_image)
adjusted_saturation = tf.image.adjust_saturation(chosen_image, saturation_factor = 2)
adjusted_brightness = tf.image.adjust_brightness(chosen_image, delta = 0.3)
adjusted_contrast = tf.image.adjust_contrast(chosen_image, contrast_factor = 2)
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=ceil(len(img_matrix_list) / ncols), ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
img_matrix_list = [chosen_image,horizontal_flipped,vertically_flipped,adjusted_saturation,adjusted_brightness,adjusted_contrast]
title_list = ["Original", "HorizontalFlipped", "VerticallyFlipped", "Saturated","Brightness","Contrast"]
plot_multiple_img(img_matrix_list, title_list, ncols = 3)
# shuffle() (if used) should be called before batch() - we want to shuffle records not batches.
train_dataset_fold_1 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_1, train_labels_fold_1))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

# Generally we don't shuffle a test/val set at all - 
# only the training set (We evaluate using the entire test set anyway, right? So why shuffle?).
# https://stackoverflow.com/questions/56944856/tensorflow-dataset-questions-about-shuffle-batch-and-repeat
# https://stackoverflow.com/questions/49915925/output-differences-when-changing-order-of-batch-shuffle-and-repeat
valid_dataset_fold_1 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_1, val_labels_fold_1))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE))
def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn

lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
rng = [i for i in range(25 if EPOCHS<25 else EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
STEPS_PER_EPOCH = train_labels_fold_1.shape[0] // BATCH_SIZE
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('GroupKFold.h5', monitor='val_loss', verbose=2, save_best_only=True)
def get_model():
    with strategy.scope():
        model = tf.keras.Sequential([
            efn.EfficientNetB3(
                input_shape=(256,256, 3),
                weights="imagenet",
                include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss = 'binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()])
    
    return model
# train_paths_fold_1 = train_fold_1.image_name.apply(format_path_train).values
# val_paths_fold_1 = val_fold_1.image_name.apply(format_path_train).values

# train_labels_fold_1 = train_fold_1.target.values
# val_labels_fold_1 = val_fold_1.target.values

# train_dataset_fold_1 = (
#     tf.data.Dataset
#     .from_tensor_slices((train_paths_fold_1, train_labels_fold_1))
#     .map(decode_image, num_parallel_calls=AUTO)
#     .map(data_augment, num_parallel_calls=AUTO)
#     .repeat()
#     .shuffle(512)
#     .batch(BATCH_SIZE)
#     .prefetch(AUTO))

# valid_dataset_fold_1 = (
#     tf.data.Dataset
#     .from_tensor_slices((val_paths_fold_1, val_labels_fold_1))
#     .map(decode_image, num_parallel_calls=AUTO)
#     .batch(BATCH_SIZE)
#     .cache()
#     .prefetch(AUTO))
model_fold_1 = get_model()
history_1 = model_fold_1.fit(train_dataset_fold_1,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_1)
probs_fold_1 = model_fold_1.predict(test_dataset,verbose = 1)
train_paths_fold_2 = train_fold_2.image_name.apply(format_path_train).values
val_paths_fold_2 = val_fold_2.image_name.apply(format_path_train).values

train_labels_fold_2 = train_fold_2.target.values
val_labels_fold_2 = val_fold_2.target.values

train_dataset_fold_2 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_2, train_labels_fold_2))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valid_dataset_fold_2 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_2, val_labels_fold_2))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))
model_fold_2 = get_model()
history_2 = model_fold_2.fit(train_dataset_fold_2,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_2)
probs_fold_2 = model_fold_2.predict(test_dataset,verbose = 1) 
train_paths_fold_3 = train_fold_3.image_name.apply(format_path_train).values
val_paths_fold_3 = val_fold_3.image_name.apply(format_path_train).values

train_labels_fold_3 = train_fold_3.target.values
val_labels_fold_3 = val_fold_3.target.values

train_dataset_fold_3 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_3, train_labels_fold_3))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valid_dataset_fold_3 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_3, val_labels_fold_3))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))
model_fold_3 = get_model()
history_3 = model_fold_3.fit(train_dataset_fold_3,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_3)
probs_fold_3 = model_fold_3.predict(test_dataset,verbose = 1) 
train_paths_fold_4 = train_fold_4.image_name.apply(format_path_train).values
val_paths_fold_4 = val_fold_4.image_name.apply(format_path_train).values

train_labels_fold_4 = train_fold_4.target.values
val_labels_fold_4 = val_fold_4.target.values

train_dataset_fold_4 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_4, train_labels_fold_4))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valid_dataset_fold_4 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_4, val_labels_fold_4))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))
model_fold_4 = get_model()
history_4 = model_fold_4.fit(train_dataset_fold_4,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_4)
probs_fold_4 = model_fold_4.predict(test_dataset,verbose = 1)
train_paths_fold_5 = train_fold_5.image_name.apply(format_path_train).values
val_paths_fold_5 = val_fold_5.image_name.apply(format_path_train).values

train_labels_fold_5 = train_fold_5.target.values
val_labels_fold_5 = val_fold_5.target.values

train_dataset_fold_5 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_5, train_labels_fold_5))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valid_dataset_fold_5 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_5, val_labels_fold_5))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))
model_fold_5 = get_model()
history_5 = model_fold_5.fit(train_dataset_fold_5,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_5)
probs_fold_5 = model_fold_5.predict(test_dataset,verbose = 1)
sample_submission = '../input/siim-isic-melanoma-classification/sample_submission.csv'
submission = pd.read_csv(sample_submission)
submission['target'] = probs_fold_5   
submission.head(20)
submission.to_csv('submission_contrast_group_k_fold_5.csv', index=False)
sample_submission = '../input/siim-isic-melanoma-classification/sample_submission.csv'
submission = pd.read_csv(sample_submission)
fold1 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_1.csv")
fold2 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_2.csv")
fold3 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_3.csv")
fold4 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_4.csv")
fold5 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_5.csv")
ensembled = (fold1['target'] + fold2['target']  + fold3['target'] + fold4['target'] + fold5['target'])/5
submission['target'] = ensembled

# ensembled = (probs_fold_1 + probs_fold_2 + probs_fold_3 + probs_fold_4 + probs_fold_5)/5
# submission['target'] = ensembled
submission.head(20)
#submitting to csv
submission.to_csv('ensembled.csv', index=False)
import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

def stratified_group_k_fold(X, y, groups, k, seed=None):
    """ https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in tqdm(sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])), total=len(groups_and_y_counts)):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices
group_by_patient_id_array = np.array(train_df['patient_id'].values)
y_labels = train_df["target"].values
skf = stratified_group_k_fold(X=train_df, y=y_labels, groups=group_by_patient_id_array, k=5, seed=42)
def get_distribution(y_vals):
        y_distr = Counter(y_vals)
        y_vals_sum = sum(y_distr.values())
        return [f'{y_distr[i] / y_vals_sum:.5%}' for i in range(np.max(y_vals) + 1)]
distrs = [get_distribution(y_labels)]
index = ['training set']

for fold_ind, (dev_ind, val_ind) in enumerate(skf, 1):
    dev_y, val_y = y_labels[dev_ind], y_labels[val_ind]
    dev_groups, val_groups = group_by_patient_id_array[dev_ind], group_by_patient_id_array[val_ind]
    # making sure that train and validation group do not overlap:
    assert len(set(dev_groups) & set(val_groups)) == 0
    
    distrs.append(get_distribution(dev_y))
    index.append(f'training set - fold {fold_ind}')
    distrs.append(get_distribution(val_y))
    index.append(f'validation set - fold {fold_ind}')

display('Distribution per class:')
pd.DataFrame(distrs, index=index, columns=[f'Label {l}' for l in range(np.max(y_labels) + 1)])
df = train_df.copy()
df['fold'] = -1
df.head()
# somehow you need to redefine this skf line here for the .loc to work
skf = stratified_group_k_fold(X=train_df, y=y_labels, groups=group_by_patient_id_array, k=5, seed=42)
for fold_number, (train_idx, val_idx) in enumerate(skf):
    df.loc[val_idx, "fold"] = fold_number
    
df.to_csv("sgkfold.csv", index=False)
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

def format_path_train(img_name):
    return GCS_PATH + '/jpeg/train/' + img_name + '.jpg'

def format_path_test(img_name):
    return GCS_PATH + '/jpeg/test/' + img_name + '.jpg'
image_size = 256

def decode_image(filename, label=None, image_size=(image_size, image_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, size = image_size) 

    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if label is None:
        return image
    else:
        return image, label
def train(fold_number):
    training_data_path = '../input/siim-isic-melanoma-classification/train/'
    df = pd.read_csv("/kaggle/working/sgkfold.csv")
    df_train = df[df.fold != fold_number].reset_index(drop=True)
    df_valid = df[df.fold == fold_number].reset_index(drop=True)
    df_train_path = df_train.image_name.apply(format_path_train).values
    df_val_path   = df_valid.image_name.apply(format_path_train).values
    df_train_labels = df_train.target.values
    df_val_labels   = df_valid.target.values
    
    AUTO = tf.data.experimental.AUTOTUNE
    # For tf.dataset
    BATCH_SIZE = 8 * strategy.num_replicas_in_sync
    EPOCHS = 3    
    
    train_dataset = (tf.data.Dataset
    .from_tensor_slices((df_train_path, df_train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))
    
    valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((df_val_path, df_val_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))
    
    def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
                   lr_min=0.00001, lr_rampup_epochs=5, 
                   lr_sustain_epochs=0, lr_exp_decay=.8):
        lr_max = lr_max * strategy.num_replicas_in_sync

        def lrfn(epoch):
            if epoch < lr_rampup_epochs:
                lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
            elif epoch < lr_rampup_epochs + lr_sustain_epochs:
                lr = lr_max
            else:
                lr = (lr_max - lr_min) *\
                     lr_exp_decay**(epoch - lr_rampup_epochs\
                                    - lr_sustain_epochs) + lr_min
            return lr
        return lrfn    
    
    lrfn = build_lrfn()
    STEPS_PER_EPOCH = df_train_labels.shape[0] // BATCH_SIZE
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('StratifiedGroupKFold.h5', monitor='val_loss', verbose=2, save_best_only=True)
    
    
    with strategy.scope():
        model = tf.keras.Sequential([
            efn.EfficientNetB3(
                input_shape=(256,256, 3),
                weights="imagenet",
                include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss = 'binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()])
    
    history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)
    return model
test_paths = test_df.image_name.apply(format_path_test).values
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 3  

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE))
fold_1 = train(0)
probs_fold_1 = fold_1.predict(test_dataset, verbose = 1)
sample_submission = '../input/siim-isic-melanoma-classification/sample_submission.csv'
submission = pd.read_csv(sample_submission)
submission['target'] = probs_fold_1  
submission.head(20)
submission.to_csv('submission_stratified_group_k_fold_1.csv', index=False)
fold_2 = train(1)
probs_fold_2 = fold_2.predict(test_dataset, verbose = 1)
fold_3 = train(2)
probs_fold_3 = fold_3.predict(test_dataset, verbose = 1)
fold_4 = train(3)
probs_fold_4 = fold_4.predict(test_dataset, verbose = 1)
fold_5 = train(4)
probs_fold_5 = fold_5.predict(test_dataset, verbose = 1)