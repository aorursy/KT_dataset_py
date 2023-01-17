!ls -lh ../input
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
n_samples = 10_000
df = pd.read_csv('../input/train_ship_segmentations.csv').dropna().sample(n_samples, random_state=34)
df.reset_index(drop=True, inplace=True)
df.head(3)
img_size = 240

def read_img(path):
    x = cv2.imread('../input/train/' + path)
    x = cv2.resize(x, (img_size, img_size))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x
def get_mask(encoded_pixels):
    mask = np.zeros((img_size, img_size), np.uint8)
    if not pd.isna(encoded_pixels):
        scale = lambda x: min(img_size-1, round(x * img_size / 768))
        rle_code = [int(i) for i in encoded_pixels.split()]
        pixels = [(scale(pixel_position % 768), scale(pixel_position // 768)) 
                     for start, length in list(zip(rle_code[0:-1:2], rle_code[1:-2:2])) 
                     for pixel_position in range(start, start + length)]
        mask[tuple(zip(*pixels))] = 1
    return mask
from joblib import Parallel, delayed

with Parallel(n_jobs=12, prefer='threads', verbose=1) as ex:
    x = ex(delayed(read_img)(e) for e in df.ImageId)
    
x = np.stack(x)
x.shape
with Parallel(n_jobs=12, prefer='threads', verbose=1) as ex:
    y = ex(delayed(get_mask)(e) for e in df.EncodedPixels)
    
y = np.stack(y)
y.shape
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, x_val.shape
def plot_img(x, y):
    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    axes[0].imshow(x)
    axes[1].imshow(y)
    for ax in axes: ax.set_axis_off()
    plt.show()
idx = np.random.choice(len(x_train))
sample_x, sample_y = x_train[idx], y_train[idx]
plot_img(sample_x, sample_y)
from keras import backend as K

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
