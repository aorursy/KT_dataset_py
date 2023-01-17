import pandas as pd

import numpy as np

import math, re, os

import random

from tqdm import tqdm

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print("Tensorflow version " + tf.__version__)

import tensorflow.keras.backend as K

import tensorflow_addons as tfa
path = '../input/cloud-anomaly-detection-images/noncloud/noncloud'

path2 = '../input/cloud-anomaly-detection-images/cloud/cloud'
all_images=[]

import os

img_list = os.listdir(path)

for i in tqdm(img_list):

    img = tf.keras.preprocessing.image.load_img(path+'/'+str(i), target_size=(384,384,3))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = img/255.

    all_images.append(img)

    

all_images= np.array(all_images)

all_images.shape
IMAGE_SIZE = [384,384]

SEED = 42

np.random.seed(SEED)

tf.random.set_seed(SEED)

n_hidden_1 = 512

n_hidden_2 = 256

n_hidden_3 = 64

n_hidden_4 = 16

n_hidden_5 = 8

convkernel = (3, 3)  # convolution kernel

poolkernel = (2, 2)  # pooling kernel
X_train, X_test = train_test_split(all_images, test_size=0.2, random_state=SEED)

print(X_train.shape, X_test.shape)
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
def get_vgg19():

    K.clear_session()

    with strategy.scope():

        image_input = tf.keras.layers.Input(shape = (*IMAGE_SIZE,3))

        vg19 = tf.keras.applications.VGG19(input_tensor = image_input, weights = 'imagenet', include_top=False)

        encoded = vg19.get_layer('block5_pool').output

        #decode

        x = tf.keras.layers.Conv2DTranspose(n_hidden_5, convkernel, strides=2, activation='relu', padding='same')(encoded)

        x = tf.keras.layers.Conv2DTranspose(n_hidden_4, convkernel, strides=2, activation='relu', padding='same')(x)

        x = tf.keras.layers.Conv2DTranspose(n_hidden_3, convkernel, strides=2, activation='relu', padding='same')(x)

        x = tf.keras.layers.Conv2DTranspose(n_hidden_2, convkernel, strides=2, activation='relu', padding='same')(x)

        x = tf.keras.layers.Conv2DTranspose(n_hidden_1, convkernel, strides=2, activation='relu', padding='same')(x)

        decoded = tf.keras.layers.Conv2DTranspose(3, convkernel, activation="sigmoid", padding='same')(x)

        model = tf.keras.models.Model(inputs = image_input, outputs = decoded)

        opt = tfa.optimizers.RectifiedAdam(lr=3e-4)

        model.compile(

            optimizer = opt,

            loss = 'mse',

            metrics = [tf.keras.metrics.RootMeanSquaredError()]

        )

        return model
model= get_vgg19()
model.load_weights('../input/model-encoder/Enc.h5')
model.summary()
cld_images=[]

import os

img_list = os.listdir(path2)

for i in tqdm(img_list):

    img = tf.keras.preprocessing.image.load_img(path2+'/'+str(i), target_size=(384,384,3))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = img/255.

    cld_images.append(img)

    

cld_images= np.array(cld_images)

cld_images.shape
n = 5

plt.figure(figsize= (20,10))



for i in range(n):

    ax = plt.subplot(2, n, i+1)

    plt.imshow(cld_images[i+50])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    ax = plt.subplot(2, n, i+1+n)

    plt.imshow(cld_images[i+30])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



plt.show()
test_predictions =  model.predict(np.concatenate((X_test[:100], cld_images), axis=0))
pred = []

for i in (tf.keras.losses.mean_squared_error(np.concatenate((X_test[:100], cld_images), axis=0), test_predictions)).numpy():

    pred.append(tf.math.reduce_mean(i).numpy())
y_cap=[]

threshold=0.001

for i in pred:

    if i> threshold:

        y_cap.append(1)

    else:

        y_cap.append(0)
pd.value_counts(y_cap)
actual = [0 for i in range(0,len(X_test[:100]))]+[1 for i in range(0,len(cld_images))]
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(actual, y_cap)

import seaborn as sns

import matplotlib.pyplot as plt     

ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 
score = pd.DataFrame()

score['matrix'] = ['F1_score','Recall','Precision']

score['Values'] = [(f1_score(actual,y_cap)),(recall_score(actual,y_cap)),(precision_score(actual,y_cap))]
score