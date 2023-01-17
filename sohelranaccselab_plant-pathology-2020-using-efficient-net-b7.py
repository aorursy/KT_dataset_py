# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

import os

import cv2

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers as L

from tensorflow.keras.utils import to_categorical



from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight

from sklearn import metrics



%matplotlib inline
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU:', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



AUTO = tf.data.experimental.AUTOTUNE

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMG_SIZE = 768



print('Batch size:', BATCH_SIZE)
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')



print(train.head())



train_path = train.image_id.apply(lambda x: f'{GCS_DS_PATH}/images/{x}.jpg').values

test_path = test.image_id.apply(lambda x: f'{GCS_DS_PATH}/images/{x}.jpg').values

train_label = train.loc[:, 'healthy':].values



# train_path, valid_path, train_label, valid_label = train_test_split(train_path, train_label,

#                                                                     test_size=0.1, stratify=train_label)
class_weight = compute_class_weight('balanced', np.unique(np.argmax(train_label, axis=1)), np.argmax(train_label, axis=1))

plt.bar(range(4), class_weight)
fig, ax = plt.subplots(2, 2)

img = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_5.jpg')

img1 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_1.jpg')

img2 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_7.jpg')

img3 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_9.jpg')

ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

ax[0, 1].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

ax[1, 0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

ax[1, 1].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
def decode_image(filename, label=None, image_size=(IMG_SIZE, IMG_SIZE)):

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

    image = tf.image.random_flip_up_down(image)

    

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.TFRecordDataset

    .from_tensor_slices((train_path, train_label))

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    .cache()

    .repeat()

    .shuffle(1024)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



# valid_dataset = (

#     tf.data.TFRecordDataset

#     .from_tensor_slices((valid_path, valid_label))

#     .map(decode_image, num_parallel_calls=AUTO)

#     .map(data_augment, num_parallel_calls=AUTO)

#     .cache()

#     .batch(BATCH_SIZE)

#     .prefetch(AUTO)

# )



test_dataset = (

    tf.data.TFRecordDataset

    .from_tensor_slices(test_path)

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

)
EPOCHS = 50

LR_START = 0.0001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.0001

LR_RAMPUP_EPOCHS = 10

LR_SUSTAIN_EPOCHS = 5

LR_EXP_DECAY = .9



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



lr = tf.keras.callbacks.LearningRateScheduler(lrfn)



y = [lrfn(x) for x in range(EPOCHS)]

plt.plot(y)
!pip install -q efficientnet
from efficientnet.tfkeras import EfficientNetB7



with strategy.scope():

    efn = EfficientNetB7(include_top=False, weights='noisy-student', pooling='max', input_shape=(IMG_SIZE, IMG_SIZE, 3))



    model = Sequential()

    model.add(efn)

    model.add(L.Dense(4, activation='softmax'))



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
mc = tf.keras.callbacks.ModelCheckpoint('weights.h5', monitor='loss', save_best_only=True, save_weights_only=True)

history = model.fit(train_dataset, epochs=EPOCHS, callbacks=[lr, mc], steps_per_epoch=train_label.shape[0] // BATCH_SIZE)
with strategy.scope():

    model.load_weights('weights.h5')

# valid_prob = model.predict(valid_dataset, verbose=1)

# print(metrics.classification_report(np.argmax(valid_label, axis=1), np.argmax(valid_prob, axis=1)))

# print(metrics.confusion_matrix(np.argmax(valid_label, axis=1), np.argmax(valid_prob, axis=1)))
probs = model.predict(test_dataset, verbose=1)

sub.loc[:, 'healthy':] = probs

sub.to_csv('submission.csv', index=False)

sub.head()
#If u like this Kernel, Upvote this Kernel