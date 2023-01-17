MODEL_NAME = "EfficientNetB0"
# MODEL_NAME = "EfficientNetB1"
# MODEL_NAME = "EfficientNetB2"
# MODEL_NAME = "EfficientNetB3"
# MODEL_NAME = "EfficientNetB4"
# MODEL_NAME = "EfficientNetB5"
# MODEL_NAME = "EfficientNetB6"
# MODEL_NAME = "EfficientNetB7"
import os
import re
import seaborn as sns
import numpy as np
import pandas as pd
import math
import random
from collections import Counter, defaultdict

from matplotlib import pyplot as plt

from sklearn import metrics

import tensorflow as tf
import tensorflow.keras.layers as L

!pip install -q efficientnet
import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets
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
AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False
def stratified_group_k_fold(X, y, groups, k, seed=None):
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
    random.Random(0).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
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
# Defining paths
KAGGLE_PATH = "/kaggle/input/siim-isic-melanoma-classification/"
IMG_PATH_TRAIN = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
IMG_PATH_TEST = "/kaggle/input/siim-isic-melanoma-classification/jpeg/test/"
GCS_PATH = KaggleDatasets().get_gcs_path("siim-isic-melanoma-classification")

# Importing train and test data
df_train = pd.read_csv(os.path.join(KAGGLE_PATH, "train.csv"))        
df_test = pd.read_csv(os.path.join(KAGGLE_PATH, "test.csv"))

# Creating columns image_path on train and test data
df_train['image_path'] = df_train['image_name'].apply(lambda x: GCS_PATH + '/jpeg/train/' + x + '.jpg').values
df_test['image_path'] = df_test['image_name'].apply(lambda x: GCS_PATH + '/jpeg/test/' + x + '.jpg').values

# Creating kfold column and train data using Stratified Group K-fold
df_train["kfold"] = -1
df_train = df_train.sample(frac=1, random_state=0).reset_index(drop=True) # shuffle dataframe
y = df_train.target.values
for fold_, (train_idx, test_idx) in enumerate(stratified_group_k_fold(df_train, df_train.target.values, df_train.patient_id.values, k=4)):
    df_train.loc[test_idx, "kfold"] = fold_
def decode_image(filename,label=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, size = (IMG_SIZE, IMG_SIZE))
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
    
def get_training_valid_dataset(kfold_i):
    train_paths_fold_i = df_train[df_train['kfold'] != kfold_i]['image_path']
    train_labels_fold_i = df_train[df_train['kfold'] != kfold_i]['target']
    
    valid_paths_fold_i = df_train[df_train['kfold'] == kfold_i]['image_path']
    valid_labels_fold_i = df_train[df_train['kfold'] == kfold_i]['target']
    
    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((train_paths_fold_i, train_labels_fold_i))
        .map(decode_image, num_parallel_calls=AUTO)
        .map(data_augment, num_parallel_calls=AUTO)
        .repeat()
        .shuffle(1024)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((valid_paths_fold_i, valid_labels_fold_i))
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )
    
    return {'train': train_dataset, 'valid': valid_dataset}


def get_test_dataset():
    test_paths = df_test['image_path']
    
    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(test_paths)
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
    )
    
    return test_dataset
LR_START = 0.00001
LR_MAX = 0.00005*strategy.num_replicas_in_sync
LR_MIN = 0.000001
LR_RAMPUP_EPOCHS = 10
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

rng = [i for i in range(50)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y);
def get_pretrained_model(model_name):
    if model_name=="EfficientNetB0":
        pretrained_model=efn.EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    if model_name=="EfficientNetB1":
        pretrained_model=efn.EfficientNetB1(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    if model_name=="EfficientNetB2":
        pretrained_model=efn.EfficientNetB2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    if model_name=="EfficientNetB3":
        pretrained_model=efn.EfficientNetB3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    if model_name=="EfficientNetB4":
        pretrained_model=efn.EfficientNetB4(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    if model_name=="EfficientNetB5":
        pretrained_model=efn.EfficientNetB5(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    if model_name=="EfficientNetB6":
        pretrained_model=efn.EfficientNetB6(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    if model_name=="EfficientNetB7":
        pretrained_model=efn.EfficientNetB7(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    return pretrained_model 
# Define model
def get_model():
    with strategy.scope():
        model = tf.keras.Sequential([
            get_pretrained_model(MODEL_NAME),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])
    
        model.compile(
            optimizer='adam',
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=[tf.keras.metrics.AUC()]
        )

    return model
IMG_SIZE = 256
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
STEPS_PER_EPOCH = df_train[df_train.kfold !=0].shape[0] // BATCH_SIZE

def train_fold(kfold_i):
    
    my_callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True),
        #tf.keras.callbacks.EarlyStopping(patience=10, monitor="val_auc", mode="max"),
        tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_NAME + "_fold_" + str((kfold_i)) + ".h5",save_best_only=True)
    ]
    
    model_i = get_model()
    dataset_i = get_training_valid_dataset(kfold_i)
    
    history_i = model_i.fit(
        dataset_i['train'],
        validation_data=dataset_i['valid'],
        epochs=10,
        callbacks=my_callbacks,
        steps_per_epoch=STEPS_PER_EPOCH
    )

    df_history_i = pd.DataFrame(history_i.history)
    df_history_i.to_csv("history_" + MODEL_NAME + "_fold_{}.csv".format(kfold_i))
def save_predictions(fold_i):
    with strategy.scope():
        model_name_fold_i = MODEL_NAME + "_fold_" + str((fold_i)) + ".h5"
        model_fold_i = tf.keras.models.load_model(model_name_fold_i)
        
        probs_fold_i = model_fold_i.predict(get_test_dataset(),verbose = 1)
        submit_fold_i = pd.read_csv(os.path.join(KAGGLE_PATH, "sample_submission.csv")) 
        submit_fold_i['target'] = probs_fold_i

        submit_fold_i.to_csv("submit_" + MODEL_NAME + "_fold_{}.csv".format(fold_i), index=False)
train_fold(0)
save_predictions(0)
#train_fold(1)
#save_predictions(1)
#train_fold(2)
#save_predictions(2)
#train_fold(3)
#save_predictions(3)