!pip install efficientnet
# ==================
# Library
# ==================
import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from efficientnet.tfkeras import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
import glob
# ==================
# Constant
# ==================
TRAIN_PATH = '../input/used-car-price-forecasting/train.csv'
TEST_PATH = '../input/used-car-price-forecasting/test.csv'
TRAIN_IMG_PATH = "../input/used-car-price-forecasting/images/train_images/"
SAVE_PATH = "train_eff0.npy"
# ===============
# Settings
# ===============
img_size = 224
batch_size = 16
# ====================
# Function
# ====================


def build_model(dim=256):
    inp = tf.keras.layers.Input(shape=(dim,dim,3))
    base = efn.EfficientNetB0(input_shape=(dim,dim,3),weights='imagenet',include_top=False) #今回はeEfficientNet B0を使います
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    return model


def load_images(img_path):
    img = load_img(img_path,target_size=(img_size,img_size))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img
# ======================
# Main
# ======================
train_df = pd.read_csv(TRAIN_PATH)
m = build_model(dim=img_size)
car_ids = train_df['id'].values
n_batches = len(car_ids) // batch_size + 1
m.summary()
features = np.zeros((len(train_df), 1280)) #modelの大きさに応じて、featuresの列数を変えてください
n = 0
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_cars = car_ids[start:end]
    batch_images = np.zeros((len(batch_cars),img_size,img_size,3))
    for i,car_id in enumerate(batch_cars):
        try:
            batch_images[i] = load_images(f"{TRAIN_IMG_PATH}{car_id}.jpg")
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,car_id in enumerate(batch_cars):
        features[n,:] = batch_preds[i]
        n += 1
np.save("train_ef0.npy",features)