config = {
    "train": {
      "lr": 0.001, 
      "optim": "Nadam",
      "epochs": 10, 
      "batch-size": 400,
      "k_batch": 40, # used in intra-enhanced triplet-loss
      "loss": "triplet-softmax", # loss function can be one of 
                                 # [triplet-softmax, large-margin-cosine-loss, 
                                 #  intra-enhanced-triplet-loss, 
                                 #  semi-hard-triplet-loss, categorical-crossentropy]
      "alpha": 0.2, # margin used in several loss functions 
      "beta": 0.1, # another margin used in intra-enhanced-triplet-loss
      "lambda_1": 0.5, # weight of anchor-center-loss in intra-enhanced-triplet-loss
      "lambda_2": 0.1, # weight of categorical-crossentropy when training with triplet-softmax
      "scale": 20, # scale used in large-margin-cosine-loss
      "reg_lambda": 0.01,# regularization weight used in large-margin-cosine-loss 
      "lr_reduce_factor": 0.5,
      "patience": 5,
      "min_lr": 1.0e-5,
      "shuffle": True,
    },
    "data": {
      "imsize": 28, 
      "imchannel": 1,
      "num_classes": 10,
      "samples_per_id": 5000, # there are 5000 samples for each class, used to order data for triplet-loss
      "val_split": 0.1 ,
    },
    "tsne": {
      "n_iter": 2500,
      "perplexity": 30, 
    },   
}

# shortcuts
train = config["train"]
data  = config["data"]
from __future__ import print_function

import os
import yaml
import argparse
import numpy as np 
import pandas as pd

from keras.utils import np_utils
from keras.datasets import mnist

class DataLoader(object):
    def __init__(self, config, one_hot = False):
        self.config = config
        self.one_hot = one_hot

    def load(self):
        data_train = pd.read_csv('../input/train.csv')
        X_data = np.array(data_train.iloc[:,1:])
        self.y_data = np.array(data_train.iloc[:,:1]).squeeze()
                
        self.input_shape = (-1, self.config["data"]["imsize"], self.config["data"]["imsize"], self.config["data"]["imchannel"])
        self.X_data = np.reshape(X_data, self.input_shape)

        if self.one_hot:
            self.y_data = np_utils.to_categorical(self.y_data, self.config["data"]["num_classes"])

        self.num_train = int(self.y_data.shape[0] * (1-self.config["data"]["val_split"]))
        self.num_val   = int(self.y_data.shape[0] * (self.config["data"]["val_split"]))

        if self.config["train"]["loss"] in ["intra-enhanced-triplet-loss", "semi-hard-triplet-loss"]: 
            print("[INFO] Ordering Data")
            self.order_data_triplet_loss()
            
        self.split_data()
        self.X_train = self.preprocess(self.X_train)
        self.X_val   = self.preprocess(self.X_val)


    def preprocess(self, data):
        data = data.astype('float32')
        return data / 255.

    def order_data_triplet_loss(self):
        data = {}
        samples_per_id = self.config["data"]["samples_per_id"]
        for label in range(self.config["data"]["num_classes"]):
            mask = self.y_data==label
            data[label] = [i for i, x in enumerate(mask) if x]
            if len(data[label]) < samples_per_id:
                data[label].extend(np.random.choice(data[label], samples_per_id - len(data[label]), replace=False))
            data[label] = data[label][:samples_per_id]

        k_batch = self.config["train"]["k_batch"]
        X_data, y_data = [], []
        for i in range(samples_per_id // k_batch):
            for label in data:
                X_data.extend(self.X_data[data[label][i*k_batch:(i+1)*k_batch]])
                y_data += [label] * k_batch

        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)

    def split_data(self):
        self.X_train = self.X_data[:self.num_train]
        self.y_train = self.y_data[:self.num_train]

        self.X_val = self.X_data[self.num_train:]
        self.y_val = self.y_data[self.num_train:]

        del self.X_data, self.y_data
                    
    def get_random_batch(self, k = 100):
        X_batch, y_batch = [], []
        for label in range(self.config["data"]["num_classes"]):
            X_mask = self.X_val[self.y_val==label]
            X_batch.extend(np.array([X_mask[np.random.choice(len(X_mask), k, replace=False)]]) if k <= len(X_mask) and k >= 0 else X_mask)
            y_batch += [label] * k if k <= len(X_mask) and k >= 0 else [label] * len(X_mask)
        X_batch = np.reshape(X_batch, self.input_shape)
        return X_batch, np.array(y_batch)
    
    def __str__(self):
        return f"# of training samples: {self.num_train} | # of validation samples: {self.num_val}"
print("[INFO] Loading Data")
dataloader = DataLoader(config)
dataloader.load()
print(dataloader)
class DataGenerator(object):
    def __init__(self, config):
        self.shuffle = config["train"]["shuffle"]
        self.batch_size = config["train"]["batch-size"]
        self.loss = config["train"]["loss"]
        self.num_classes = config["data"]["num_classes"]

    def generate(self, X, y):
        ''' Generates batches of samples '''
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(len(y))
            # Generate batches
            batches = np.arange(len(indexes)//self.batch_size)
            if not self.shuffle: np.random.shuffle(batches)

            for batch in batches:
                # Find list of ids
                batch_indecies = indexes[batch*self.batch_size:(batch+1)*self.batch_size]
                if self.loss == "triplet-softmax":
                    y_1 = y[batch_indecies]
                    y_2 = np_utils.to_categorical(y_1, self.num_classes)
                    yield X[batch_indecies], [y_1, y_2]
                else:
                    yield X[batch_indecies], y[batch_indecies]

    def __get_exploration_order(self, data_size):
        ''' Generates order of exploration '''
        idxs = np.arange(data_size)
        if self.shuffle == True:
            np.random.shuffle(idxs)
        return idxs   
print("[INFO] Creating Generators")
train_gen = DataGenerator(config).generate(dataloader.X_train, dataloader.y_train)
val_gen = DataGenerator(config).generate(dataloader.X_val, dataloader.y_val)
from __future__ import print_function

import os
import yaml
import argparse

import tensorflow as tf
import keras.backend as K 
import keras.optimizers as optimizers

from keras import losses
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Input, GlobalAveragePooling2D, LeakyReLU, SeparableConv2D, BatchNormalization, Add
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.callbacks import ReduceLROnPlateau

def get_model(input_shape, config, top = True):
    input_img = Input(input_shape)
    num_classes = config["data"]["num_classes"]

    def __body(input_img):
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_img)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        embedding = Dense(128, activation='relu')(x)
        return embedding

    def __head(embedding):
        x   = Dropout(0.5)(embedding)
        out = Dense(num_classes, activation='softmax')(x)
        return out

    x = __body(input_img)
    if config["train"]["loss"] in ["triplet-softmax"] and top:
        y = __head(x)
        model = Model(inputs=input_img, outputs=[x, y])
    else:
        if top: x = __head(x)
        model = Model(inputs=input_img, outputs=x)
    return model
print("[INFO] Building Model")
input_shape = (data["imsize"], data["imsize"], data["imchannel"])
model = get_model(input_shape, config, top=train["loss"] in ["categorical-crossentropy", "triplet-softmax"])
model.summary()
import functools
import numpy as np
import tensorflow as tf
from keras import backend as K

def get_loss_function(func):
    return {
        'triplet-softmax': ([semi_hard_triplet_loss(config["train"]["alpha"]), 'categorical_crossentropy'],  [1, train["lambda_2"]]),
        'large-margin-cosine-loss': (large_margin_cos_loss(config["train"]), None),
        'intra-enhanced-triplet-loss': (intra_enhanced_triplet_loss(config["train"]), None),
        'semi-hard-triplet-loss': (semi_hard_triplet_loss(config["train"]["alpha"]), None),
        'categorical-crossentropy': (losses.categorical_crossentropy, None),
    }.get(func, (losses.categorical_crossentropy, None))

def __anchor_center_loss(embeddings, margin, batch_size = 400, k = 40):
    """Computes the anchor-center loss
    Minimizes intra-class distances. Assumes embeddings are ordered 
    such that every k samples belong to the same class, where the 
    number of classes is batch_size // k.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: intra-class distances should be within this margin
        batch_size: number of embeddings 
        k: number of samples per class in embeddings
    Returns:
        loss: scalar tensor containing the anchor-center loss
    """
    loss = tf.constant(0, dtype='float32')
    for i in range(0,batch_size,k):
        anchors = embeddings[i:i+k] 
        center = tf.reduce_mean(anchors, 0)
        loss = tf.add(loss, tf.reduce_sum(tf.maximum(tf.reduce_sum(tf.square(anchors - center), axis=1) - margin, 0.)))
    return tf.reduce_mean(loss)

def __semi_hard_triplet_loss(labels, embeddings, margin = 0.2):
    return tf.contrib.losses.metric_learning.triplet_semihard_loss(labels[:,0], embeddings, margin=margin)

def __intra_enhanced_triplet_loss(labels, embeddings, lambda_1, alpha, beta, batch_size, k):
    return tf.add(__semi_hard_triplet_loss(labels, embeddings, alpha), tf.multiply(lambda_1, __anchor_center_loss(embeddings, beta, batch_size, k)))

def __large_margin_cos_loss(labels, embeddings, alpha, scale, regularization_lambda, num_cls = 10):
    num_features = embeddings.get_shape()[1]
    
    with tf.variable_scope('centers_scope', reuse = tf.AUTO_REUSE):
        weights = tf.get_variable("centers", [num_features, num_cls], dtype=tf.float32, 
                initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(1e-4), trainable=True)

    embedds_feat_norm = tf.nn.l2_normalize(embeddings, 1, 1e-10)
    weights_feat_norm = tf.nn.l2_normalize(weights, 0, 1e-10)

    xw_norm = tf.matmul(embedds_feat_norm, weights_feat_norm)
    margin_xw_norm = xw_norm - alpha

    labels = tf.squeeze(tf.cast(labels, tf.int32))
    label_onehot = tf.one_hot(labels, num_cls)
    value = scale*tf.where(tf.equal(label_onehot, 1), margin_xw_norm, xw_norm)

    cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=value))

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cos_loss = cos_loss + regularization_lambda * tf.add_n(regularization_losses)
    return cos_loss 

def semi_hard_triplet_loss(margin):
    @functools.wraps(__semi_hard_triplet_loss)
    def loss(labels, embeddings):
        return __semi_hard_triplet_loss(labels, embeddings, margin)
    return loss

def intra_enhanced_triplet_loss(config):
    @functools.wraps(__intra_enhanced_triplet_loss)
    def loss(labels, embeddings):
        return __intra_enhanced_triplet_loss(labels, embeddings, config["lambda_1"], config["alpha"], config["beta"], config["batch-size"], config["k_batch"])
    return loss

def large_margin_cos_loss(config):
    @functools.wraps(__large_margin_cos_loss)
    def loss(labels, embeddings):
        return __large_margin_cos_loss(labels, embeddings, config["alpha"], config["scale"], config["reg_lambda"])
    return loss
loss_func, loss_weights = get_loss_function(train["loss"])
optim = getattr(optimizers, train["optim"])(train["lr"])
model.compile(loss=loss_func, loss_weights=loss_weights, optimizer=optim, metrics=[])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=train["lr_reduce_factor"], patience=train["patience"], min_lr=train["min_lr"])
print("[INFO] Start Training")
model.fit_generator(
    generator = train_gen,
    steps_per_epoch = dataloader.num_train//train["batch-size"],
    validation_data = val_gen,
    validation_steps= dataloader.num_val//train["batch-size"],
    shuffle = False,
    workers = 0,
    epochs = train["epochs"],
    callbacks=[reduce_lr]
)
import seaborn as sns
import matplotlib.pyplot as plt

def scatter(x, labels, config, run_title = "MNIST"):
    palette = np.array(sns.color_palette("hls", config["data"]["num_classes"]))

    fig, ax = plt.subplots()
    ax.scatter(x[:,0], x[:,1], lw=0, s=40, alpha=0.2, c=palette[labels.astype(np.int)])

    for idx in range(config["data"]["num_classes"]):
        xtext, ytext = np.median(x[labels == idx, :], axis=0)
        txt = ax.text(xtext, ytext, str(idx), fontsize=20)

    plt.title(f"{run_title} T-SNE")
    plt.show()
from sklearn.manifold import TSNE

def visualize(embeddings, run_title = "MNIST"):
    tsne = TSNE(n_components=2, perplexity=config["tsne"]["perplexity"], verbose=1, n_iter=config["tsne"]["n_iter"])
    tsne_embeds = tsne.fit_transform(embeddings)
    scatter(tsne_embeds, y_batch, config, run_title)
X_batch, y_batch = dataloader.get_random_batch(k = -1)
embeddings = X_batch.reshape(-1, 784) 
visualize(embeddings, "Pixels")
X_batch, y_batch = dataloader.get_random_batch(k = -1)
embeddings = model.predict(X_batch, batch_size=config["train"]["batch-size"], verbose=1)
if train["loss"] in ["triplet-softmax"]:
    embeddings = embeddings[0]
visualize(embeddings)