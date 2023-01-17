! conda install -y hvplot=0.5.2 bokeh==1.4.0
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from toolz.curried import map, pipe, compose_left, partial
from typing import Union, Tuple, List, Dict
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
from abc import ABCMeta
from itertools import chain
from operator import add
import holoviews as hv
import pandas as pd
import hvplot.pandas
from sklearn.datasets import load_digits, load_boston
import tensorflow as tf
import tensorflow_probability as tfp
from functools import reduce

hv.extension('bokeh')

data = load_boston()
EPOCHS = 1000

class FFNN(tf.keras.Model):
    def __init__(self, layers = (4, )):
        super(FFNN, self).__init__()
        
        self.inputs = tf.keras.layers.InputLayer((3, 3))
        self.dense = list(map(lambda units: tf.keras.layers.Dense(units, activation='selu'), layers))
        self.final = tf.keras.layers.Dense(1, activation='linear')
        
    def call(self, inputs):
        
        return reduce(lambda x, f: f(x), [inputs, self.inputs, *self.dense, self.final])
    
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        
        loss = tf.keras.losses.mse(predictions, label)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
pd.Series(data.target).hvplot.kde(xlabel='Log-Target Value')
train_ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(data.data.astype('float32')),
                                               tf.convert_to_tensor(np.log(data.target.astype('float32'))))).batch(32)

model = FFNN()
model.compile(loss='mse')

optimizer = tf.keras.optimizers.Adam()
for epoch in range(EPOCHS):
    for sample in train_ds:
        inputs, label = sample
        gradients = train_step(inputs, label)
        
y_pred = model(data.data.astype('float32')).numpy()

model.summary()
class Explainer(tf.keras.Model):
    def __init__(self, 
                 input_shape: Tuple = (569, 30),
                 target_shape: Tuple = (569, 1), 
                 alpha = 1e-3,
                 l1_ratio = 1):
        super(Explainer, self).__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        self.feature_importances = tf.keras.layers.Embedding(input_shape[0]+1, 
                                                             input_shape[1],
                                                             embeddings_initializer = 'normal',
                                                             embeddings_regularizer = tf.keras.regularizers.l1_l2(l1=alpha*l1_ratio,
                                                                                                             l2=alpha*(1-l1_ratio)))
        self.tranpose = tf.keras.layers.Lambda(lambda x: tf.transpose(x))
        self.dot = tf.keras.layers.Dot((0,0))
        
    def call(self, X: Dict[str, tf.Tensor]):
        weights = self.feature_importances(X['index'])
        transpose_weights = self.tranpose(weights)
        return (X['features'] @ transpose_weights)
@tf.function
def explainer_loss(y_true, y_pred, sample_weights=None):
    length = tf.shape(y_true)[0]
    square_error = tf.square(y_pred - (y_true * tf.ones((length, length), 'float32')))
    if sample_weights is not None:
        return tf.reduce_mean(square_error*tf.stop_gradient(sample_weights))
    return tf.reduce_mean(square_error)
kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(1,X.shape[1] * 0.75)
y = y_pred.astype(np.float32)
X = np.c_[data.data.astype(np.float32), np.ones(shape=(data.data.shape[0], 1), dtype=np.float32)]

lime = Explainer(input_shape=X.shape, target_shape=y.shape)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

index =  tf.reshape(tf.range(X.shape[0]), (-1,))
train = tf.data.Dataset.from_tensor_slices((X, index, y)).batch(X.shape[0]//3)

@tf.function
def lime_train_step(inputs, model, explainer, optimizer):
    with tf.GradientTape() as tape:        
        sample_weights = kernel.matrix(inputs['features'], inputs['features'])
        y_pred = model(inputs['features'][:, :-1])
        loss = explainer_loss(explainer(inputs), y_pred, sample_weights)
    gradients = tape.gradient(loss, explainer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, explainer.trainable_variables))
    
    return loss

for epoch in range(EPOCHS):
    for sample in train:
        X_sample, index_sample, y_sample = sample
        lime_train_step({'features': X_sample, 'index': index_sample}, model, lime, optimizer)
(pd.DataFrame(lime.feature_importances(tf.slice(tf.random.shuffle(index), [0], (1,))).numpy()[:, :-1], columns=data.feature_names, index=['Feature'])
 .hvplot.bar(title='LIME Example Explaination')
 .opts(xrotation=45))
(pd.DataFrame(lime.feature_importances(index).numpy()[:, :-1], columns=data.feature_names)
 .pipe(lambda df: df.mean(0) / df.std(0))
 .hvplot.bar(title='LIME Average Effect')
 .opts(xrotation=45))
@tf.function
def factorial(x):
    return tf.exp(tf.math.lgamma(x))

@tf.function
def choose(n, k):
    return factorial(n) / (factorial(k) * factorial(n-k))

@tf.function
def shapely_kernel(z_prime):
    M = tf.dtypes.cast(tf.shape(z_prime)[1], z_prime.dtype)
    z_abs = tf.reshape(tf.reduce_sum(z_prime, -1), (-1,))
    return tf.math.divide_no_nan((M-1), choose(M, z_abs) * z_abs * (M - z_abs))
shap = Explainer(input_shape=X.shape, target_shape=y.shape, alpha=0.)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

index =  tf.reshape(tf.range(X.shape[0]), (-1,))
train = tf.data.Dataset.from_tensor_slices((X, index, y)).batch(X.shape[0]//3)

@tf.function
def shap_train_step(inputs, model, explainer, optimizer):
    with tf.GradientTape() as tape:
        z_prime = tf.dtypes.cast(tf.random.uniform(tf.shape(inputs['features']))>0.5, 'float32')
        z_prime_minus = tf.abs(z_prime-1)
        
        sample_weights = shapely_kernel(z_prime)
        
        z_mean = tf.reduce_mean(inputs['features'])
        z = inputs['features'] * z_prime + z_mean * z_prime_minus
        y_pred = model(inputs['features'][:, :-1])
        
        loss = explainer_loss(explainer({'features': z, 'index': index_sample}), y_pred, sample_weights)
    gradients = tape.gradient(loss, explainer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, explainer.trainable_variables))
    
    return loss

for epoch in range(EPOCHS*10):
    losses = []
    for sample in train:
        X_sample, index_sample, y_sample = sample
        inputs = {'features': X_sample, 'index': index_sample}
        losses.append(shap_train_step(inputs, model, shap, optimizer).numpy())
(pd.DataFrame(shap.feature_importances(tf.slice(tf.random.shuffle(index), [0], (1,))).numpy()[:, :-1], columns=data.feature_names, index=['Feature'])
 .hvplot.bar(title='SHAP Example Explaination')
 .opts(xrotation=45))
(pd.DataFrame(shap.feature_importances(index).numpy()[:, :-1], columns=data.feature_names)
 .pipe(lambda df: df.mean(0))
 .hvplot.bar(title='SHAP Average Effect')
 .opts(xrotation=45))