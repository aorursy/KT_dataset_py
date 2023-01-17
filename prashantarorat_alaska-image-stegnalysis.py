from kaggle_datasets import KaggleDatasets

import tensorflow as tf
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

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

EPOCHS = 10

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
def append_path(pre):

    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))
import numpy as np

import os



for folder in os.listdir("../input/alaska2-image-steganalysis/"):

    if os.path.isfile("../input/alaska2-image-steganalysis/"+folder):

        continue

        

    train_filenames = np.array(os.listdir(f"/kaggle/input/alaska2-image-steganalysis/{folder}"))

    paths = append_path(f"{folder}")(train_filenames)

    np.save(f"{folder}.npy",paths)
from tqdm.notebook import tqdm

import pandas as pd

import gc
path = "../input/alaska2-image-steganalysis/"



classes_index = {

    "Cover":0,

    "JMiPOD":1,

    "JUNIWARD":2,

    "UERD":3

}



data = []



for folder in os.listdir(path):

    if folder == "Test" or os.path.isfile(os.path.join(path,folder)):

        continue

    folder_path = os.path.join(path,folder)

    class_ = classes_index[folder] 

    print(f"working on {folder}")

    

    for image in tqdm(os.listdir(folder_path)):

        image_path = os.path.join(folder_path,image)

        data.append([image_path,class_])

    print(f"Completed Class {folder} \n")

    

train = pd.DataFrame(data,columns=['image','class'])

del data

gc.collect()
train.to_csv("train_data.csv",index=None)