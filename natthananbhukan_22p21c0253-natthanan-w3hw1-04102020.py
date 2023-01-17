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
# For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

BATCH_SIZE = 4 * strategy.num_replicas_in_sync



#GOBAL Variable

TRAIN_PATH = '../input/super-ai-image-classification/train/train'

TEST_PATH = '../input/super-ai-image-classification/val/val'



LR = 0.001

EPOCHS = 100

WARMUP = 10

IMG_SIZE_h = 224

IMG_SIZE_w = 224
df_train = pd.read_csv(os.path.join(TRAIN_PATH,'train.csv'))
df_train.head()
df_train['category'].value_counts().to_frame().reset_index().plot(kind='bar')
def show_train_img(df, cat):

    

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))

    ten_random_samples = df[df['category'] == cat]['id'].sample(10).values

    

    for idx, image in enumerate(ten_random_samples):

        final_path = os.path.join(TRAIN_PATH, 'images',image)

        img = cv2.imread(final_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes.ravel()[idx].imshow(img)

        axes.ravel()[idx].axis('off')

    plt.tight_layout()
show_train_img(df_train,0)
show_train_img(df_train,0)
show_train_img(df_train,1)
show_train_img(df_train,1)
def joinPathTrain(path):

    new_path = os.path.join(GCS_DS_PATH, 'train', 'train', 'images', path)

    return new_path
df_train['path'] = df_train['id'].apply(joinPathTrain)

df_train['category'] = df_train['category'].apply(float)
df_train.head(10)
def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.flip_up_down(image)

    image = tf.image.random_crop(image, size=(200,200), seed=2020)

    image = tf.image.random_brightness(image, max_delta=0.25)

    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.image.random_hue(image, max_delta=0.2)

    

    if label is None:

        return image

    else:

        return image, label
def decode_image(filename, label=None, image_size=(300, 300) , file =True):

    if file:

        filename = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(filename, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        return image, label
data_path_list = np.array(df_train['path'].to_list())

data_label_list = np.array(df_train['category'].to_list())
from sklearn.utils import class_weight



class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(data_label_list),

                                                 data_label_list)
from sklearn.model_selection import train_test_split



train_paths, valid_paths, train_labels, valid_labels = train_test_split(data_path_list, 

                                                                        data_label_list, 

                                                                        stratify=data_label_list,

                                                                        test_size=0.1, 

                                                                        random_state=2020)



train_paths.shape, valid_paths.shape, train_labels.shape, valid_labels.shape
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((valid_paths, valid_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)
from keras import backend as K



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
import tensorflow_addons as tfa



Focal_loss = tfa.losses.SigmoidFocalCrossEntropy()
!pip install -q efficientnet
from tensorflow.keras.layers import Dense, Lambda, Input

from tensorflow.keras.models import Model

from efficientnet.tfkeras import EfficientNetB5
 %%time

input_tensor = Input(shape=(IMG_SIZE_h, IMG_SIZE_w, 3))

optimizer = tf.keras.optimizers.Adam(LR)



with strategy.scope():

     model_effnet = tf.keras.Sequential([

         EfficientNetB5(weights='imagenet', # imagenet

                        include_top=False,

                        pooling='avg',

                        input_tensor=input_tensor,

                        input_shape=(IMG_SIZE_h, IMG_SIZE_w, 3)

                        ),

         Dense(8, activation='relu'),

         Dense(1, activation='sigmoid')

     ])

    

     model_effnet.layers[0].trainable = False

     

     model_effnet.compile(optimizer = optimizer,

                   loss=Focal_loss, 

                   metrics=['acc',f1_m])

    

     model_effnet.summary()
import math



def get_cosine_schedule_with_warmup(lr, num_warmup_steps, num_training_steps, num_cycles=0.5):

    def lrfn(epoch):

        if epoch < num_warmup_steps:

            return float(epoch) / float(max(1, num_warmup_steps)) * lr

        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr



    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



lr_schedule= get_cosine_schedule_with_warmup(lr=LR, num_warmup_steps=WARMUP, num_training_steps=EPOCHS)
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint



n_steps = train_paths.shape[0] // BATCH_SIZE

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

class_weight_dict = {

                        0:class_weights[0],

                        1:class_weights[1]

                    }





history_effnet = model_effnet.fit(

    train_dataset, 

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS,

    class_weight = class_weight_dict,

    callbacks=[lr_schedule,es],

    verbose = 1

)
# Get training and test loss histories

training_loss = history_effnet.history['loss']

test_loss = history_effnet.history['val_loss']



# Create count of the number of epochs

epoch_count = range(1, len(training_loss) + 1)



# Visualize loss history

plt.plot(epoch_count, training_loss, 'r--')

plt.plot(epoch_count, test_loss, 'b-')

plt.legend(['Training Loss', 'Test Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()
# Get training and test loss histories

training_acc = history_effnet.history['acc']

test_acc = history_effnet.history['val_acc']



# Create count of the number of epochs

epoch_count = range(1, len(training_loss) + 1)



# Visualize loss history

plt.plot(epoch_count, training_acc, 'r--')

plt.plot(epoch_count, test_acc, 'b-')

plt.legend(['Training Acc', 'Test Acc'])

plt.xlabel('Epoch')

plt.ylabel('Acc')

plt.show()
test_image_list = []



for image_file in os.listdir(os.path.join(TEST_PATH,'images')):

  dict_image = {

                "id":image_file,

                "category": np.random.randint(2, size=1)[0]

                }

  test_image_list.append(dict_image)



df_test = pd.DataFrame(test_image_list)
def addPathTest(value):

  path = os.path.join(GCS_DS_PATH, 'val', 'val','images',value)

  return path
df_test['path'] = df_test['id'].apply(addPathTest)
df_test['category'] = df_test['category'].apply(str)
df_test.head()
df_test.shape
test_paths = df_test['path'].to_list()
def data_augment_test(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=0.25)

    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

#     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

#     image = tf.image.random_hue(image, max_delta=0.2)

    

    if label is None:

        return image

    else:

        return image, label
test_dataset_tta = (

        tf.data.Dataset

        .from_tensor_slices(test_paths)

        .map(decode_image, num_parallel_calls=AUTO)

        .cache()

        .map(data_augment_test, num_parallel_calls=AUTO)

        .batch(BATCH_SIZE)

)
tta_times = 5

probabilities = []



for i in range(tta_times+1):

    print('TTA Number: ', i, '\n')

    probabilities.append(model_effnet.predict(test_dataset_tta, verbose=1))

    

tta_pred = np.mean(probabilities, axis=0)
df_test['category'] = tta_pred.round()
df_test['category'] = df_test['category'].apply(int)
df_test = df_test.drop(['path'],axis=1)
df_test = df_test.set_index('id')
df_test
df_test['category'].value_counts()
df_test.to_csv('submission.csv')
model_effnet.save_weights("model.h5")