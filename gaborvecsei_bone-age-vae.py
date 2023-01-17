import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm_notebook
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model

import matplotlib.pyplot as plt
%matplotlib inline
BASE_INPUT_FOLDER_PATH = Path("../input")
df = pd.read_csv(BASE_INPUT_FOLDER_PATH / "boneage-training-dataset.csv", index_col=0, header=0)
train_image_paths = list((BASE_INPUT_FOLDER_PATH / "boneage-training-dataset" / "boneage-training-dataset").glob("*.png"))

image_paths_df = pd.DataFrame()
image_paths_df["id"] = np.array([x.stem for x in train_image_paths]).astype(int)
image_paths_df["path"] = [str(x) for x in train_image_paths]
image_paths_df.set_index("id", inplace=True)

df = df.merge(image_paths_df, left_index=True, right_index=True)
df.head(2)
def read_images(image_paths:list):
    train_images = []
    train_images_flat = []

    for p in image_paths:
        image = cv2.imread(str(p), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (64, 64))

        train_images.append(image)
        train_images_flat.append(image.flatten())

    train_images = np.array(train_images)
    train_images_flat = np.array(train_images_flat)
    
    return train_images, train_images_flat
images, images_flat = read_images(df["path"].values)
VAE_INPUT_HEIGHT = 64
VAE_INPUT_WIDTH = 64
VAE_INPUT_DIM = (VAE_INPUT_HEIGHT, VAE_INPUT_WIDTH, 3)
VAE_Z_DIM = 32
VAE_DENSE_SIZE = 1024
VAE_BATCH_SIZE = 32

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], VAE_Z_DIM), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


class VAE():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.z_dim = output_dim

        self.model, self.encoder, self.decoder = self._build()

    @classmethod
    def init_default(cls):
        obj = cls(VAE_INPUT_DIM, VAE_Z_DIM)
        return obj

    def _build(self):
        vae_x = Input(shape=self.input_dim)
        vae_c1 = Conv2D(filters=32, kernel_size=4, strides=2, activation="relu")(vae_x)
        vae_c2 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(vae_c1)
        vae_c3 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(vae_c2)
        vae_c4 = Conv2D(filters=128, kernel_size=4, strides=2, activation="relu")(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(self.z_dim)(vae_z_in)
        vae_z_log_var = Dense(self.z_dim)(vae_z_in)

        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
        vae_z_input = Input(shape=(self.z_dim,))

        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1, 1, VAE_DENSE_SIZE))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")
        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3 = Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation="relu")
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4 = Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation="sigmoid")
        vae_d4_model = vae_d4(vae_d3_model)

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        def vae_r_loss(y_true, y_pred):
            return K.sum(K.square(y_true - y_pred), axis=[1, 2, 3])

        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.sum(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

        vae.compile(optimizer='rmsprop', loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

        return vae, vae_encoder, vae_decoder

    def train(self, data, epochs=32, include_callbacks=True):
        callbacks_list = []

        if include_callbacks:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
            callbacks_list = [early_stopping]

        self.model.fit(data, data,
                       shuffle=True,
                       epochs=epochs,
                       batch_size=VAE_BATCH_SIZE,
                       validation_split=0.2,
                       callbacks=callbacks_list)
images_float = images.astype(np.float32) / 255.0
vae = VAE.init_default()
vae.train(images_float, epochs=200)
fig = plt.figure(figsize=(30, 5))

n_rows = 2
n_cols = 20

for i in range(0, n_cols):
    image_float = images_float[i]
    encoded_image = vae.encoder.predict(np.expand_dims(image_float, 0))
    decoded_image = vae.decoder.predict(encoded_image)[0]
    reconstructed_image = (decoded_image*255).astype(np.uint8)
    
    ax_rec = fig.add_subplot(n_rows, n_cols, i+1)
    ax_rec.imshow(reconstructed_image)
    ax_rec.set_xticks([])
    ax_rec.set_yticks([])
    
    ax_ori = fig.add_subplot(n_rows, n_cols, n_cols+1+i)
    ax_ori.imshow(images[i])
    ax_ori.set_xticks([])
    ax_ori.set_yticks([])