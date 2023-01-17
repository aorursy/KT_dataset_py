! conda install -y hvplot=0.5.2 bokeh==1.4.0
! conda install -y -c conda-forge sklearn-contrib-py-earth
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
from functools import reduce

hv.extension('bokeh')
data = load_boston()
print(data.DESCR)
EPOCHS = 50

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
from sklearn import tree
import matplotlib.pyplot as plt
clf = tree.DecisionTreeRegressor(max_depth=4, min_weight_fraction_leaf=0.15)
clf = clf.fit(data.data, y_pred)
plt.figure(figsize=(30,7))
dot_data = tree.plot_tree(clf, max_depth=4, fontsize=12, feature_names=data.feature_names, filled=True, rounded=True)
from pyearth import Earth
earth = Earth(max_degree=2, allow_linear=True, feature_importance_type='gcv')

earth.fit(data.data, y_pred, xlabels=data.feature_names.tolist())
print(earth.summary())
print(earth.summary_feature_importances())
(pd.DataFrame([earth._feature_importances_dict['gcv']], columns=data.feature_names, index=['Features'])
 .hvplot.bar(title="'Non-linear' Linear Model Global Approximation Feature Importances")
 .opts(xrotation=45, ylabel='Importance'))