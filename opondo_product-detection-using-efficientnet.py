import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from kaggle_datasets import KaggleDatasets
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [16, 8]

print('Using Tensorflow version:', tf.__version__)
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
# !ls /kaggle/input
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path('shopee-product-detection-student')

# Configuration
EPOCHS = 32
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
train_df = pd.read_csv('../input/after-cleaned/image_cleaned.csv')
test_df = pd.read_csv('../input/shopee-product-detection-student/test.csv')

train_df.shape, test_df.shape
train_df.head()
def show_train_img(category):
    
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
    
    train_path = '/kaggle/input/shopee-product-detection-open/train/train/train/'
    ten_random_samples = pd.Series(os.listdir(os.path.join(train_path, category))).sample(10).values
    
    for idx, image in enumerate(ten_random_samples):
        final_path = os.path.join(train_path, category, image)
        img = cv2.imread(final_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes.ravel()[idx].imshow(img)
        axes.ravel()[idx].axis('off')
    plt.tight_layout()
def show_test_img():
    
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
    
    test_path = '/kaggle/input/shopee-product-detection-open/test/test/test/'
    ten_random_samples = pd.Series(os.listdir(test_path)).sample(10).values
    
    for idx, image in enumerate(ten_random_samples):
        final_path = os.path.join(test_path, image)
        img = cv2.imread(final_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes.ravel()[idx].imshow(img)
        axes.ravel()[idx].axis('off')
    plt.tight_layout()
# pick random samples

dataset_path = {}

categories = np.sort(train_df['category_2'].unique())

for cat in categories:
    try:
        dataset_path[cat] = train_df[train_df['category_2'] == cat]['filename'].sample(2400)
    except:
        dataset_path[cat] = train_df[train_df['category_2'] == cat]['filename'].sample(frac=1.)
dataset_path[0]
category_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
                 '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                 '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                 '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                 '40', '41']
train_paths = []

for idx, key in enumerate(dataset_path.keys()):
    if key == idx:
        for path in dataset_path[idx]:
            train_paths.append(os.path.join(GCS_DS_PATH, 'train', 'train', 'train', category_list[idx], path))
len(train_paths)
labels = []

for label in dataset_path.keys():
    labels.extend([label] * len(dataset_path[label]))
    
len(labels)
from tensorflow.keras.utils import to_categorical

# convert to numpy array
train_paths = np.array(train_paths)

# convert to one-hot-encoding-labels
train_labels = to_categorical(labels)
# from sklearn.model_selection import train_test_split

# train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, 
#                                                                         train_labels, 
#                                                                         stratify=train_labels,
#                                                                         test_size=0.1, 
#                                                                         random_state=2020)

# train_paths.shape, valid_paths.shape, train_labels.shape, valid_labels.shape
test_paths = []

for path in test_df['filename']:
    test_paths.append(os.path.join(GCS_DS_PATH,  'test', 'test', 'test', path))
    
test_paths = np.array(test_paths)
def decode_image(filename, label=None, image_size=(380, 380)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
#     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
#     image = tf.image.random_hue(image, max_delta=0.2)
    
    if label is None:
        return image
    else:
        return image, label
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .cache()
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# valid_dataset = (
#     tf.data.Dataset
#     .from_tensor_slices((valid_paths, valid_labels))
#     .map(decode_image, num_parallel_calls=AUTO)
#     .batch(BATCH_SIZE)
#     .cache()
#     .prefetch(AUTO)
# )

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)
%%time
!pip install -q efficientnet
from tensorflow.keras.layers import Dense
from efficientnet.tfkeras import EfficientNetB4
import keras.backend as K

def categorical_focal_loss_with_label_smoothing(gamma=2.0, alpha=0.25,ls=0.1, classes=42.0):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
        y_ls = (1 - α) * y_hot + α / classes
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
        ls    -- label smoothing parameter(alpha)
        classes     -- No. of classes
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
        ls    -- 0.1
        classes     -- 4
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        #label smoothing
        y_pred_ls = (1 - ls) * y_pred + ls / classes
        # Clip the prediction value
        y_pred_ls = K.clip(y_pred_ls, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred_ls)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred_ls), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss
%%time

with strategy.scope():
    model = tf.keras.Sequential([
        EfficientNetB4(weights='imagenet', # noisy-student
                       include_top=False,
                       pooling='avg'), # max
        Dense(42, activation='softmax')
    ])
    
    model.layers[0].trainable = False
    
    model.compile(optimizer = 'adam',
                  loss = categorical_focal_loss_with_label_smoothing(gamma=2.0, alpha=0.75, ls=0.125, classes=42.0), # num classes
                  metrics=['accuracy'])
    
    model.summary()
n_steps = train_labels.shape[0] // BATCH_SIZE  # 86428 / 128 = 675

history = model.fit(
    train_dataset, 
    steps_per_epoch=n_steps,
    # validation_data=valid_dataset,
    epochs=2,
)
# model.save()
# Get training and test loss histories
training_loss = history.history['loss']
# test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
test_dataset_tta = (
        tf.data.Dataset
        .from_tensor_slices(test_paths)
        .map(decode_image, num_parallel_calls=AUTO)
        .cache()
        .map(data_augment, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
)

tta_times = 5
probabilities = []

for i in range(tta_times+1):
    print('TTA Number: ', i, '\n')
    probabilities.append(model.predict(test_dataset_tta, verbose=1))
    
tta_pred = np.mean(probabilities, axis=0)
# change with prediction
test_df['category'] = tta_pred.argmax(axis=1)

# then add zero-padding
test_df['category'] = test_df['category'].apply(lambda x: str(x).zfill(2))

test_df.to_csv('sub_with_tta_2.csv', index=False)
