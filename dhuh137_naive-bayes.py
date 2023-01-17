import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from fastai.vision import Path

from tqdm import tqdm

import PIL

import gc
path_data = Path("../input/data/data")

path_data.ls()
vocab = ['alarm' , 'lock', 'movie', 'rain', 'weather']

dim = 224

X = np.array([])

for cat in tqdm(vocab):

    path_tmp = path_data/cat

    imgs = path_tmp.ls()

    cat_imgs = []

    for im in imgs:

        cat_imgs.append(np.array(

            [np.array(PIL.Image.open(i).resize((dim,dim)))/255 for i in im.ls()]))

    cat_imgs = np.array(cat_imgs)

    X = np.append(X,cat_imgs)



X.shape
from sklearn.decomposition import PCA

X_pca = []

pca = PCA(224)

max_idx = 0

X_pca_tmp = []

for seq in tqdm(X):

    seq_t = []

    for im in seq:

        pca.fit(im.reshape(224,224*3))

        tmp_idx = np.where(np.cumsum(pca.explained_variance_ratio_)>0.98)[0][0]

        if max_idx < tmp_idx:

            max_idx = tmp_idx

        seq_t.append(np.array(pca.singular_values_))

    X_pca_tmp.append(np.array(seq_t))



min_seq = min([i.shape[0] for i in X])

pca = PCA(max_idx if max_idx<min_seq else min_seq)

for seq in tqdm(X_pca_tmp):

    pca.fit(seq)

    X_pca.append(np.array(pca.singular_values_))





X_pca = np.array(X_pca)

X_pca.shape
X_pca = np.array(X_pca)
from keras.preprocessing.sequence import pad_sequences

pad_length = max([seq.shape[0] for seq in X_pca])

X_padded = pad_sequences(X_pca, pad_length, 'float64', 'post')

X_padded.shape
Y = np.array([np.argmax(np.array(pd.get_dummies(vocab).iloc[u//10])) for u in range(50)])

Y.shape
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.3, shuffle=True)
from sklearn.naive_bayes import *

from sklearn.svm import SVC, NuSVC

cnb = ComplementNB()

mnb = MultinomialNB()

bnb = BernoulliNB()

nsvc = NuSVC()

svc = SVC()
cnb.fit(X_train, y_train)

mnb.fit(X_train, y_train)

bnb.fit(X_train, y_train)

nsvc.fit(X_train, y_train)

svc.fit(X_train, y_train)



cnb.score(X_test,y_test),mnb.score(X_test,y_test),bnb.score(X_test,y_test),svc.score(X_test,y_test),nsvc.score(X_test,y_test)
cnb.fit(X_pca, Y)

mnb.fit(X_pca, Y)

bnb.fit(X_pca, Y)

nsvc.fit(X_pca, Y)

svc.fit(X_pca, Y)

cnb.score(X_pca,Y),mnb.score(X_pca,Y),bnb.score(X_pca,Y),svc.score(X_pca,Y),nsvc.score(X_pca,Y)
import time

from IPython.display import clear_output

import matplotlib.pyplot as plt

import random

seq=random.randint(0,49)

for idx in range(len(X[seq])):

    plt.imshow(X[seq][idx].reshape(224,224,3))

    plt.title('''Actual:{}

                \nMNB:{}\nCNB:{}\nBNB:{}

                \nSVC:{}\nNSVC:{}'''.format(vocab[Y[seq]],

                                            vocab[mnb.predict(X_pca[seq].reshape(1, -1))[0]],

                                            vocab[cnb.predict(X_pca[seq].reshape(1, -1))[0]],

                                            vocab[bnb.predict(X_pca[seq].reshape(1, -1))[0]],

                                            vocab[svc.predict(X_pca[seq].reshape(1, -1))[0]],

                                            vocab[nsvc.predict(X_pca[seq].reshape(1, -1))[0]]))

    plt.show()

    time.sleep(0.1)

    plt.close()

    clear_output(wait=True)