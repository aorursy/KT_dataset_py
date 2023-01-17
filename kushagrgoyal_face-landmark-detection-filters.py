import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras as k
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import optuna

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

path = '../input/face-images-with-marked-landmark-points/'
def plot_image_landmarks(img_array, df_landmarks, index):
    plt.imshow(img_array[index, :, :, 0], cmap = 'gray')
    plt.scatter(df_landmarks.iloc[index][0: -1: 2], df_landmarks.iloc[index][1: : 2], c = 'y')
    plt.show()
def gaussian_k(x0, y0, sigma, width, height):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float) ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def create_heatmaps(landmarks, height, width, sigma):
    heatmaps = np.zeros(shape = (len(landmarks), height, width, len(landmarks.columns)//2))
    for i, val in enumerate(landmarks.index.values):
        temp = landmarks.iloc[i].values
        
        all_x = temp[0::2]
        all_y = temp[1::2]

        temp_landmarks = list(zip(all_x, all_y))
        
        for j, land in enumerate(temp_landmarks):
            heatmaps[i, :, :, j] = gaussian_k(land[0], land[1], sigma, width, height)
    return np.around(heatmaps, 5)

def visualize_heatmaps(index, heatmaps):
    heat = heatmaps[index]
    plt.figure(figsize = (15, 8))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        sns.heatmap(heat[:, :, i])
    plt.show()
x = np.load(path + 'face_images.npz')
x = x.get(x.files[0])
x = np.moveaxis(x, -1, 0)
x = x.reshape(x.shape[0], x.shape[1], x.shape[1], 1)
df = pd.read_csv(path + 'facial_keypoints.csv')
df.head()
df_new = df.dropna()
# df_new = df_new / 96

x_new = x[df_new.index.values, :, :, :]
x_new = x_new / 255

df_new.reset_index(inplace = True, drop = True)
plot_image_landmarks(x_new, df_new, 10)
x_train, x_test, y_train, y_test = train_test_split(x_new, df_new, test_size = 0.1, random_state = 0)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
y_train_heatmaps = create_heatmaps(y_train, 96, 96, 5)
y_test_heatmaps = create_heatmaps(y_test, 96, 96, 5)
visualize_heatmaps(10, y_train_heatmaps)
def create_model():
    ins = k.Input(shape = x_new.shape[1:])
    
    x = k.layers.Conv2D(128, 7, activation = 'relu', strides = 2)(ins)
    x = k.layers.BatchNormalization()(x)
    
    for i in range(6):
        x = k.layers.Conv2D(128, 5, activation = 'relu')(x)
        x = k.layers.Conv2D(128, 5, activation = 'relu')(x)
        x = k.layers.BatchNormalization()(x)
    
#     for i in range(5):
#         x = k.layers.Conv2D(128, 7, activation = 'relu')(x)
#         x = k.layers.BatchNormalization()(x)
    
#     for i in range(4):
#         x = k.layers.Conv2D(128, 5, activation = 'relu')(x)
#         x = k.layers.BatchNormalization()(x)
    
    outs = k.layers.Conv2D(30, 1)(x)
    
    model = k.Model(inputs = ins, outputs = outs)
    
    opt = k.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = opt, loss = 'mse')
    return model

def create_model_heatmaps():
    ins = k.Input(shape = x_train.shape[1:])
    
    x = k.layers.Conv2D(128, 7, activation = 'relu', padding = 'same')(ins)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.Conv2D(128, 5, activation = 'relu', padding = 'same')(x)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.MaxPool2D(3)(x)
    
#     x = k.layers.Conv2DTranspose(128, 5, activation = 'relu')(x)
#     x = k.layers.BatchNormalization()(x)
    
#     x = k.layers.Conv2DTranspose(128, 5, activation = 'relu')(x)
#     x = k.layers.BatchNormalization()(x)
    
    outs = k.layers.Conv2DTranspose(15, 3, strides = 3, activation = 'sigmoid')(x)
    
    model = k.Model(inputs = ins, outputs = outs)
    opt = k.optimizers.Adam()
    model.compile(loss = 'mse', optimizer = opt)
    return model

def create_model_heatmaps_1():
    ins = k.Input(shape = x_train.shape[1:])
    
    x = k.layers.Conv2D(256, 7, activation = 'relu', padding = 'same')(ins)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.Conv2D(256, 5, activation = 'relu', padding = 'same')(x)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.Conv2D(256, 3, activation = 'relu')(x)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.Conv2D(128, 3, activation = 'relu')(x)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.Conv2D(128, 3, activation = 'relu')(x)
    x = k.layers.Conv2D(128, 3, activation = 'relu')(x)
    x = k.layers.Conv2D(128, 3, activation = 'relu')(x)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.MaxPool2D(3)(x)
    
    x = k.layers.Conv2DTranspose(128, 3, activation = 'relu')(x)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.Conv2DTranspose(128, 3, activation = 'relu')(x)
    x = k.layers.BatchNormalization()(x)
    
    outs = k.layers.Conv2DTranspose(15, 3, strides = 3, activation = 'sigmoid')(x)
    
    model = k.Model(inputs = ins, outputs = outs)
    opt = k.optimizers.Adam(learning_rate = 0.01)
    model.compile(loss = 'mse', optimizer = opt, metrics = 'mse')
    return model
model1 = create_model_heatmaps_1()
model1.summary()
calls = [k.callbacks.ReduceLROnPlateau(patience = 5, verbose = 1), k.callbacks.EarlyStopping(patience = 10)]
history1 = model1.fit(x = x_train, y = y_train_heatmaps, batch_size = 32, epochs = 100, verbose = 2, validation_data = (x_test, y_test_heatmaps), callbacks = calls)
model = create_model_heatmaps()
model.summary()
calls = [k.callbacks.ReduceLROnPlateau(patience = 2), k.callbacks.EarlyStopping(patience = 10)]
history = model.fit(x = x_train, y = y_train_heatmaps, batch_size = 32, epochs = 150, validation_data = (x_test, y_test_heatmaps), callbacks = calls)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'][3:])
y_pred = model1.predict(x_test)
# y_pred = np.around(y_pred, 5)
# y_pred = y_pred * 96
# sns.heatmap(y_pred[0, :, :, 3])
# sns.heatmap(y_test_heatmaps[0, :, :, 0])

plt.figure(figsize = (20, 10))
for i in range(15):
    plt.subplot(3, 5, i+1)
    sns.heatmap(y_pred[4, :, :, i], cmap = 'Spectral_r')
plt.show()
def find_xy(heatmap, n):
    top_n = np.sort(heatmap.flatten())[-n:]
    top_n_locs = [np.where(heatmap == i) for i in top_n]
    top_n_locs = np.array([np.array([i[0][0], i[1][0]]) for i in top_n_locs])
    norm_top_n = top_n / np.max(top_n)
    
    y = np.sum(top_n_locs[:, 0] * norm_top_n)/np.sum(norm_top_n)
    x = np.sum(top_n_locs[:, 1] * norm_top_n)/np.sum(norm_top_n)
    
    return np.array([x, y])

def get_landmarks(predictions, n):
    landmarks = []
    
    for i, val in enumerate(predictions):
        lands = []
        for j in range(val.shape[-1]):
            xy = find_xy(predictions[i, :, :, j], n=6)
            lands.append(xy)
        landmarks.append(np.array(lands))
    landmarks = np.array(landmarks)
    landmarks = landmarks.reshape(landmarks.shape[0], -1)
    return landmarks
landmarks_pred = get_landmarks(y_pred, 5)
landmarks_pred
def plot_img_preds(images, truth, pred, index):
    plt.imshow(images[index, :, :, 0], cmap = 'gray')
    
    t = np.array(truth)[index]
    plt.scatter(t[0::2], t[1::2], c = 'y')
    
    p = pred[index, :]
    plt.scatter(p[0::2], p[1::2], c = 'r')
    
    plt.show()
plot_img_preds(x_test, y_test, landmarks_pred, 7)