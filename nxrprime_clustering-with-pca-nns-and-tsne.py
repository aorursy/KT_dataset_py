import numpy as np, pandas as pd

import warnings; warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import tensorflow as tf
train = pd.read_csv('../input/lish-moa/train_features.csv')

targs = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

y = targs[targs.columns[55]].values
pca_ = PCA(n_components=2)

pca = pca_.fit_transform(train.drop(["sig_id", 'cp_type', 'cp_time', 'cp_dose'], axis=1))

pca_t = pca_.fit_transform(test.drop(["sig_id"], axis=1))

tsne_ = TSNE(n_components=2)

tsne = tsne_.fit_transform(train.drop(["sig_id", 'cp_type', 'cp_time', 'cp_dose'], axis=1))

print('Explained variance for PCA', pca_.explained_variance_ratio_.sum())
fig = plt.figure(figsize=(10, 10));colors=['green', 'red']

plt.axis('off')

for color, i, ax, option in zip(colors, [0, 1], [121, 122], [pca, tsne]):

    plt.scatter(tsne[y == i, 0], tsne[y == i, 1], color=color, s=1,

                alpha=.8, marker='.')
fig, axs = plt.subplots(1, 2, figsize=(20, 10));colors=['green', 'red']

for color, i, ax, option in zip(colors, [0, 1], [121, 122], [pca, pca_t]):

    axs[0].scatter(pca[y == i, 0], pca[y == i, 1], color=color, s=1,

                alpha=.8, marker='.')

    axs[1].scatter(pca_t[y == i, 0], pca_t[y == i, 1], color=color, s=1,

                alpha=.8, marker='.')
def create():

    model = tf.keras.Sequential([

    tf.keras.layers.Input(2),

    tf.keras.layers.Dense(128),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Activation("relu"),

    tf.keras.layers.Dense(512),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(400),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(206, activation="sigmoid")

    ])

    model.compile(optimizer=tf.optimizers.Adam(),

                  loss='binary_crossentropy', 

                  )

    return model

model = create()

model.fit(tsne, targs.drop(["sig_id"], axis=1).values.astype(float), epochs=8, verbose=False)

preds = model.predict(pca_t)

fig = plt.figure(figsize=(9, 9));colors=['green', 'red']

for color, i, ax, option in zip(colors, [0, 1], [121, 122], [pca, tsne]):

    plt.scatter(preds[y == i, 0], preds[y == i, 1], color=color, s=1,

                alpha=.8, marker='.')