! conda install -y hvplot=0.5.2 bokeh==1.4.0
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
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
from functools import reduce
from sklearn.inspection import permutation_importance, plot_partial_dependence, partial_dependence

hv.extension('bokeh')
data = load_boston()
print(data.DESCR)
pd.Series(data.target).hvplot.kde(xlabel='Log-Target Value')
estimator = MLPRegressor((4,)) 

X, y = data.data, np.log(data.target)

estimator.fit(X, y)
y_pred = estimator.predict(X)
(pd.Series(y - y_pred).hvplot.kde(xlabel='Model Errors', title='MLP Model') +\
pd.Series(y - y_pred).hvplot.box(ylabel='').opts(invert_axes=True, height=100)).cols(1)
plot_partial_dependence(estimator, X, [(1,2), 2, 1], feature_names=data.feature_names, n_cols=2)
imp = permutation_importance(estimator, X, y, scoring=None, n_repeats=1000, n_jobs=-1)

(pd.DataFrame(imp['importances'].T, columns=data.feature_names)
 .melt(var_name='Feature', value_name='Importance')
 .hvplot.violin(y='Importance', by='Feature'))
EPOCHS = 50

class FFNN(tf.keras.Model):
    def __init__(self, layers = (4, )):
        super(FFNN, self).__init__()
        
        self.inputs = tf.keras.layers.InputLayer((3, 3))
        self.dense = list(map(lambda units: tf.keras.layers.Dense(units, activation='relu'), layers))
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
def sensitivity_importance(X: tf.Tensor, 
                                reference: tf.Tensor, 
                                model: tf.keras.Model,
                                sample=1000):
    """
    """
    length = tf.shape(X)[0]
    features = tf.shape(X)[1]
    all_subs = tf.dtypes.cast(tf.random.uniform((sample, features)) > 0.5, 'float32')
    
    f_mean = model(reference)[0]
    
    count_subs = tf.shape(all_subs)[0]
    
    @tf.function
    def apply(x):
        return tf.reduce_mean(model(tf.where(all_subs==1, 
                                     tf.ones((count_subs,features))*x, 
                                     tf.ones((count_subs,features))*reference_point)) - f_mean, axis=1)
    
    all_sub_float = tf.dtypes.cast(all_subs, 'float32')
    return tf.map_fn(apply, X) @ all_sub_float / tf.reduce_sum(all_sub_float, axis=0)

reference_point = data.data.mean(0).reshape(1, -1)
X = data.data.astype('float32')
%%timeit
permutive_values = sensitivity_importance(X, reference_point, model)
permutive_values = sensitivity_importance(X, reference_point, model)
pd.Series(permutive_values.numpy().mean(0) - permutive_values.numpy().mean(), index=data.feature_names).hvplot.bar(title='Average Sensitivity')