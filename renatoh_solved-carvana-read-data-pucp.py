import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
df = pd.read_csv('../input/train_masks.csv', usecols=['img'])
df['masks'] = '../input/train_masks/' + df.img.str.replace('.jpg', '_mask.gif')
df['img'] = '../input/train/' + df.img
print(df.shape)
df.head()
img_size = 256

def read_img(path):
    x = cv2.imread(path)
    x = cv2.resize(x, (img_size, img_size))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x

def read_mask(path):
    x = Image.open(path)
    x = x.resize([img_size, img_size])
    return np.asarray(x)
from joblib import Parallel, delayed

with Parallel(n_jobs=12, prefer='threads', verbose=1) as ex:
    x = ex(delayed(read_img)(e) for e in df.img)
    
x = np.stack(x)
x.shape
with Parallel(n_jobs=12, prefer='threads', verbose=1) as ex:
    y = ex(delayed(read_mask)(e) for e in df.masks)
    
y = np.stack(y)[..., None]
y.shape
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, x_val.shape
def plot_img(x, y):
    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    axes[0].imshow(x)
    axes[1].imshow(y[:,:,0])
    for ax in axes: ax.set_axis_off()
    plt.show()
idx = np.random.choice(len(x_train))
sample_x, sample_y = x_train[idx], y_train[idx]
plot_img(sample_x, sample_y)
import keras.backend as K
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

# Ej:
# model.compile(Adam(lr=1e-3), bce_dice_loss, metrics=['accuracy', dice_coef])
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, add

layers = {}
n_layers = 4
nf = 16

m_in = Input((img_size, img_size, 3))
m_x = BatchNormalization()(m_in)
layers[0] = Conv2D(nf, 3, padding='same')(m_x)

def downsample(x): return Conv2D(nf, 3, strides=(2,2), padding='same', activation='relu')(x)
def upsample(x): return Conv2DTranspose(nf, 3, strides=(2,2), padding='same', activation='relu')(x)

for i in range(n_layers):
    nf *= 2
    layers[i+1] = downsample(layers[i])
else: m_x = layers[i+1]

for i in range(n_layers)[::-1]:
    nf //= 2
    m_x = add([upsample(m_x), layers[i]])
    
m_out = Conv2D(1, 1, padding='same', activation='sigmoid')(m_x)

model = Model(m_in, m_out)
model.compile('adam', bce_dice_loss, metrics=['accuracy', dice_coef])
model.summary()
bs = 128
log = model.fit(x_train, y_train, bs, epochs=5, validation_data=[x_val, y_val])
K.eval(model.optimizer.lr.assign(1e-4))
log = model.fit(x_train, y_train, bs, epochs=5, validation_data=[x_val, y_val])
idx = np.random.choice(len(x_val))
sample_x, sample_y = x_val[idx], y_val[idx]
sample_pred = model.predict(sample_x[None])[0] > 0.5
plot_img(sample_x, sample_y)
plot_img(sample_x, sample_pred)


