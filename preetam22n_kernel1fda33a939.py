import os
import random
import numpy as np

import tensorflow as tf
from keras.layers import Conv2D, Lambda, Dense, Flatten, MaxPooling2D, concatenate, BatchNormalization, Dropout
from keras.models import Input, Model, Sequential
from keras.regularizers import l2
from keras import backend as K
tf.config.list_physical_devices('GPU')
def read_data(fname):
    imgData = np.load(fname)
    return imgData
def get_triplet(cls_pos, cls_neg):
    anc = np.zeros([15, 100000, 1], dtype=np.float16)
    pos = np.zeros([15, 100000, 1], dtype=np.float16)
    neg = np.zeros([15, 100000, 1], dtype=np.float16)
    loc = ['../input/arrhythmia-dataset/Train/Normal/',
           '../input/arrhythmia-dataset/Train/Sup-Ventricular/',
           '../input/arrhythmia-dataset/Train/Ventricular/',
           '../input/arrhythmia-dataset/Train/Ventricular+Sup-Ventricular/']
    i = cls_pos
    j = i
    k = cls_neg
    files1 = os.listdir(loc[i])
    files2 = os.listdir(loc[k])
    idx1 = np.random.randint(len(files1))
    idx2 = np.random.randint(len(files1))
    while idx1==idx2:
        idx2 = np.random.randint(len(files1))
    idx3 = np.random.randint(len(files2))
    img1 = read_data(loc[i]+files1[idx1])
    img2 = read_data(loc[i]+files1[idx2])
    img3 = read_data(loc[k]+files2[idx3])
    #print(loc[i]+files1[idx1])
    #print(loc[i]+files1[idx2])
    #print(loc[k]+files2[idx3])
    anc[:, :, 0] = img1
    pos[:, :, 0] = img2
    neg[:, :, 0] = img3
    return anc, pos, neg
def get_triplet_batch(batch_size, cls_pos, cls_neg):
        anchor_image = []
        positive_image = []
        negative_image = []
        for i in range(batch_size):
            ida, idp, idn = get_triplet(cls_pos, cls_neg)
            anchor_image.append(ida)
            positive_image.append(idp)
            negative_image.append(idn)

        ai = np.array(anchor_image)
        pi = np.array(positive_image)
        ni = np.array(negative_image)
        return [ai, pi, ni]
batch_size = 2
cls_pos, cls_neg = 0, 1
anchor_image, positive_image, negative_image = get_triplet_batch(batch_size, cls_pos, cls_neg)
anchor_image.shape
emb_size = 200
def make_model():
    embedding_model = Sequential()
    embedding_model.add(Conv2D(4, (4,25), strides=(2,10), padding='valid', activation='relu', input_shape=(15,100000,1)))
    embedding_model.add(MaxPooling2D(1,2))
    embedding_model.add(Dropout(0.5))
    embedding_model.add(Conv2D(8, (3,10), strides=(2,5), padding='valid', activation='relu'))
    embedding_model.add(MaxPooling2D(1,2))
    embedding_model.add(Dropout(0.5))
    embedding_model.add(Conv2D(16, (1,3), activation='relu'))
    embedding_model.add(Flatten())
    embedding_model.add(Dense(emb_size, activation='sigmoid'))
    embedding_model.add(Lambda(lambda x:tf.keras.backend.l2_normalize(x, axis=1)))
    return embedding_model
def triplet_loss(alpha, emb_dim):
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
        distance1 = tf.keras.losses.cosine_similarity(anc, pos)
        distance2 = tf.keras.losses.cosine_similarity(anc, neg)
        return tf.keras.backend.clip(distance1 - distance2 + alpha, 0., None)
    return loss
def data_generator(batch_size, pos, neg, emb_size):
    while True:
        x = get_triplet_batch(batch_size, pos, neg)
        y = np.zeros((batch_size, 3*emb_size))
        yield x,y
def make_network():
    embedding_model = make_model()
    
    in_anc = Input(shape=(15,100000,1))
    in_pos = Input(shape=(15,100000,1))
    in_neg = Input(shape=(15,100000,1))

    em_anc = embedding_model(in_anc)
    em_pos = embedding_model(in_pos)
    em_neg = embedding_model(in_neg)

    out = concatenate([em_anc, em_pos, em_neg], axis=1)

    siamese_net = Model([in_anc, in_pos, in_neg], out)
    
    return siamese_net
