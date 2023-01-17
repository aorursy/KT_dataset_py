import numpy as np
import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

from itertools import product

import tensorflow as tf
DP_DIR = '../input/shuffle-csvs/'
INPUT_DIR = '../input/quickdraw-doodle-recognition/'

BASE_SIZE = 256
NCSVS = 100
NCATS = 340

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)
import ast

def to_triplets(drawing):
    last_x, last_y = 0, 0
    triplets = []
    for stroke in drawing:
        xs = stroke[0]
        ys = stroke[1]
        first_triplet = [[xs[0] - last_x, ys[0] - last_y, 1]]
        stroke_triplets = list(zip(np.diff(xs), np.diff(ys), np.repeat([0], len(xs) - 1)))
        new_triplets = first_triplet + stroke_triplets
        triplets.extend(new_triplets)
        last_x, last_y = xs[-1], ys[-1]
    return np.array(triplets)

def to_strokes(triplets, th = 0.5):
    strokes_triplets = np.array_split(triplets, np.where(triplets[:, 2] > th)[0])
    if len(strokes_triplets[0]) == 0:
        strokes_triplets = strokes_triplets[1:]
    drawing = []
    last_x, last_y = 0, 0
    for stroke_triplets in strokes_triplets:
        xs = (np.cumsum(stroke_triplets[:, 0]) + last_x).tolist()
        ys = (np.cumsum(stroke_triplets[:, 1]) + last_y).tolist()
        last_x, last_y = xs[-1], ys[-1]
        drawing.append([xs, ys])
    return drawing

owls = pd.read_csv(INPUT_DIR + 'train_simplified/owl.csv', nrows=256)
owls['drawing'] = owls['drawing'].apply(ast.literal_eval)
owls.head()
import os 
import json
from keras.preprocessing.sequence import pad_sequences

def image_generator_xd(batchsize, ks):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                x = df_to_image_array_xd(df,  add_y=True)
                yield x

def df_to_image_array_xd(df, add_y=False):
    df['drawing_val'] = df['drawing'].apply(json.loads)
    drawing = df['drawing_val'].apply(to_triplets)
    inp = drawing.tolist()
    inp = pad_sequences(inp, padding='post', truncating='post', dtype=np.float32, maxlen=150)
    inp[:, :, :2] /= 256.
    df.loc[df.countrycode.isnull(), 'countrycode'] = 'Nan'
    if add_y:
        y = keras.utils.to_categorical(df.y, num_classes=NCATS)
        return inp, y
    return inp
def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
from keras.engine.topology import Layer

from tensorflow.python.framework import function

@function.Defun(tf.float32, tf.float32)
def norm_grad(x, dy):
    return dy*(x/(tf.norm(x, axis=0) + 1e-6))

@function.Defun(tf.float32, grad_func=norm_grad)
def norm(x):
    return tf.norm(x, axis=0)

@function.Defun(tf.float32, tf.float32)
def dist_neg_grad(x, dy):
    d = tf.sqrt(x[0, :] ** 2 - x[1, :] ** 2) + 1e-6
    return dy * tf.stack([x[0, :] / d, -x[1, :] / d])

@function.Defun(tf.float32, grad_func=dist_neg_grad)
def dist_neg(x):
    return tf.sqrt(x[0, :] ** 2 - x[1, :] ** 2)

def div_no_nan(x, y):
    return tf.where(y == 0, tf.zeros_like(x), tf.div(x, y))

size = 32
image_size = 64

class NeuralRasterizationLayer(Layer):
    def __init__(self, render_size=32, parallel_iterations=32, **kwargs):
        self.size = render_size
        self.parallel_iterations = parallel_iterations
        self.support_mask = True
        super().__init__(**kwargs)
        
    def compute_mask(self, x, mask=None):
        return None
    
    def compute_mask(self, input_shape, input_mask=None):
        return None
    
    def build(self, input_shape):
        self.output_dim = (self.size, self.size, 1)
        indices_np = np.array(list(product(np.arange(self.size),np.arange(self.size)))).reshape((self.size,self.size,2))
        self.tf_indices_np = tf.constant(indices_np, dtype=tf.float32)
        
    def call(self, x, mask=None):
        assert isinstance(x, list)
        points, atts = x
        inp = tf.concat([points, atts], axis=-1)
        images = self.batch_draw_lines(inp)
        return tf.reshape(images, (-1, self.size, self.size, 1))
    
    def tf_point_line_dist(self, points, p0, p1):
        points0 = points[:, 0]
        points1 = points[:, 1]
        p00 = p0[:, 0]
        p01 = p0[:, 1]
        p10 = p1[:, 0]
        p11 = p1[:, 1]
        d1, d2 = p10-p00, p11-p01
        enum = tf.abs(d1*points1-d2*points0+p11*p00-p10*p01)
        denom = norm(tf.stack([d1, d2]))
        denom = tf.reshape(denom, (-1,))
        val = div_no_nan(enum, (denom + 1e-6))
        val0 = tf.abs(points0 - p00)
        pred0 = tf.cast(tf.equal(p00 - p10, 0), tf.float32)
        val1 = tf.abs(points1 - p01)
        pred1 = tf.cast(tf.equal(p01 - p11, 0), tf.float32)
        return pred0 * val0 + pred1 * val1 + (1 - pred0 - pred1) * val

    def tf_point_point_dist(self, points, p, width=1.0):
        return norm(tf.stack([points[:, 0] - p[:, 0], points[:, 1] - p[:, 1]]))

    def draw_lines2(self, points, width=0.5):
        points_x = tf.cumsum(points[:, 1]) * self.size
        points_y = tf.cumsum(points[:, 0]) * self.size
        points0 = tf.stack([points_x[:-1], points_y[:-1], points[:-1, 3], points[:-1, 2]], axis=1)
        points1 = tf.stack([points_x[1:], points_y[1:], points[1:, 3], points[1:, 2]], axis=1)

        points0 = tf.expand_dims(points0, 1)
        points1 = tf.expand_dims(points1, 1)
        points0_tile = tf.tile(points0, [1, self.size * self.size, 1])
        points1_tile = tf.tile(points1, [1, self.size * self.size, 1])
        points0_tile = tf.reshape(points0_tile, ((-1, 4)))
        points1_tile = tf.reshape(points1_tile, ((-1, 4)))
        indices = tf.reshape(self.tf_indices_np, (-1,2))
        indices = tf.tile(indices, [tf.shape(points0)[0], 1])

        cond_mask1 = tf.not_equal(points1_tile[:, 0], 0.0)
        cond_mask2 = tf.not_equal(points1_tile[:, 1], 0.0)
        cond_mask3 = tf.not_equal(points0_tile[:, 0], 0.0)
        cond_mask4 = tf.not_equal(points0_tile[:, 1], 0.0)
        cond0 = tf.equal(points1_tile[:, 3], 0.0)
        cond1 = tf.less_equal(indices[:, 0], points1_tile[:, 0] + width)
        cond2 = tf.greater_equal(indices[:, 0], points0_tile[:, 0] -  width)
        cond3 = tf.less_equal(indices[:, 1], points1_tile[:, 1] +  width)
        cond4 = tf.greater_equal(indices[:, 1], points0_tile[:, 1] - width)
        cond1b = tf.greater_equal(indices[:, 0], points1_tile[:, 0] - width)
        cond2b = tf.less_equal(indices[:, 0], points0_tile[:, 0] +  width)
        cond3b = tf.greater_equal(indices[:, 1], points1_tile[:, 1] -  width)
        cond4b = tf.less_equal(indices[:, 1], points0_tile[:, 1] + width)
        cond = cond0 & (((cond1 & cond2) | (cond1b & cond2b)) & ((cond3 & cond4) | (cond3b & cond4b)))
        cond = cond & ((cond_mask1 & cond_mask2) | (cond_mask3 & cond_mask4))

        ind = tf.where(cond)
        indices = tf.gather_nd(indices, ind)
        points0_tile = tf.gather_nd(points0_tile, ind)
        points1_tile = tf.gather_nd(points1_tile, ind)

        dist = self.tf_point_line_dist(indices, points0_tile, points1_tile)
        dist_p0 = self.tf_point_point_dist(indices, points0_tile, width)
        dist_p1 = self.tf_point_point_dist(indices, points1_tile, width)
        l0 = dist_neg(tf.stack([dist_p0, dist]))
        l1 = dist_neg(tf.stack([dist_p1, dist]))
        l0 = tf.reshape(l0, (-1,))
        l1 = tf.reshape(l1, (-1,))
        l0 = tf.where(l0 > 0, l0, tf.zeros_like(l0))
        l1 = tf.where(l1 > 0, l1, tf.zeros_like(l1))
        lam0, lam1 = div_no_nan(l0, (l0 + l1 + 1e-6)),  div_no_nan(l1, (l0 + l1 + 1e-6))
        points = tf.where(dist < width, tf.ones_like(dist), tf.zeros_like(dist))
        points *= tf.cast(lam0 * points0_tile[:, 2] + lam1 * points1_tile[:, 2], tf.float32)

        points_all = tf.scatter_nd(ind, points, (tf.cast(tf.shape(cond)[0], dtype=tf.int64),))

        images = tf.reshape(points_all, ((-1, size, size)))
        image = tf.reduce_max(images, axis=0)
        image = tf.minimum(1.0, image)
        image *= 2.0
        image -= 1.0
        return image

    def batch_draw_lines(self, points_batch, width=1):
        return tf.map_fn(self.draw_lines2, points_batch, parallel_iterations=32)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.size, self.size, 1)
from keras.layers import Lambda

Resize = Lambda(lambda x: tf.image.resize_images(x, (image_size, image_size)))
import keras

from keras.layers import Input, Concatenate, Dense, BatchNormalization, Conv2D, Activation, Add, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.activations import relu
from keras.models import Model
from keras.optimizers import Adam

def ConvBnRelu(x, filters, kernel_size, strides, shortcut=False):
	inp = x
	x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	if shortcut:
		inp = Conv2D(filters, 1, strides=strides, use_bias=False)(inp)
		inp = BatchNormalization()(inp)
		inp = Activation('relu')(inp)
		x = Add()([x, inp])
	return x

def SketchNetMini(input_shape):
	inp = Input(shape=input_shape) # 64
	x = ConvBnRelu(inp, 16, 5, 2) # 32
	x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x) # 16
	x = ConvBnRelu(x, 32, 5, 1) # 16
	x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x) # 8
	x = ConvBnRelu(x, 64, 3, 1, shortcut=True) # 8
	x = ConvBnRelu(x, 64, 3, 1, shortcut=True) # 8
	x = ConvBnRelu(x, 64, 3, 1, shortcut=True) # 8
	x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x) # 4
	x = ConvBnRelu(x, 128, 3, 1)
	x = ConvBnRelu(x, 256, 1, 1)
	x = GlobalAveragePooling2D()(x)
	x = Dropout(rate=0.01)(x)
	model = Model(inp, x)
	return model
import keras
from keras.layers import InputLayer, LSTM, CuDNNLSTM, Bidirectional, TimeDistributed, Input, Dense, Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Dropout, Conv1D, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy, top_k_categorical_accuracy

rnn_inp = Input(shape=(150, 3))
net = BatchNormalization()(rnn_inp)
net = Conv1D(32, (5,), activation = 'relu', padding='same')(net)
net = Conv1D(64, (5,), activation = 'relu', padding='same')(net)
rnn = Bidirectional(CuDNNLSTM(64, return_sequences=True))(net)
att = TimeDistributed(Dense(1, activation='sigmoid', bias_initializer=keras.initializers.zeros()))(rnn)
img = NeuralRasterizationLayer(render_size=size)([rnn_inp, att])
img = Resize(img)
cnn_model = SketchNetMini((image_size, image_size, 1))
image_out = cnn_model(img)
pred_class = Dense(NCATS, activation='softmax', name='class')(image_out)
model_img = Model(rnn_inp, img)
model = Model(rnn_inp, pred_class)
print(model.summary())
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=1024)
x_valid = df_to_image_array_xd(valid_df)
n = 10
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n*n):
    ax = axs[i % n, i // n]
    images = model_img.predict(x_valid[i].reshape((1, -1, 3)))
    images += 1.0
    images /= 2.0
    ax.imshow(images[0].reshape((image_size,image_size)), cmap='gray')
plt.show()
train_datagen = image_generator_xd(batchsize=32, ks=range(NCSVS - 1))
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=3200)
valid_data = df_to_image_array_xd(valid_df, add_y=True)
model.compile(optimizer=Adam(lr=5e-4), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
hist = model.fit_generator(
    train_datagen, steps_per_epoch=1000, epochs=10, verbose=1, 
    validation_data=valid_data
)
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
hist = model.fit_generator(
    train_datagen, steps_per_epoch=1000, epochs=10, verbose=1, 
    validation_data=valid_data
)
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=1024)
x_valid = df_to_image_array_xd(valid_df)
n = 10
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n*n):
    ax = axs[i % n, i // n]
    images = model_img.predict(x_valid[i].reshape((1, -1, 3)))
    images += 1.0
    images /= 2.0
    ax.imshow(images[0].reshape((image_size,image_size)), cmap='gray')
plt.show()