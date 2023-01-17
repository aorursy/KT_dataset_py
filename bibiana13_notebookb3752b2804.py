import os

import pandas as pd

from os.path import join

from scipy.fftpack import fft

from scipy.io import wavfile

import numpy as np

import matplotlib.pyplot as plt

import IPython.display as ipd
train_audio_path = '../input/train/audio/'
samples = dict()

for label in os.listdir(train_audio_path):

    samples[label] = [f for f in os.listdir(join(train_audio_path, label)) if f.endswith('.wav')]



infosamples = pd.DataFrame(data = [[label, len(samp)] for label, samp in samples.items()], columns=['label', 'num_samples'])    

infosamples
labels = infosamples.label[infosamples.label != '_background_noise_'].values

labels
def custom_fft(y, fs):

    T = 1.0 / fs

    N = y.shape[0]

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    vals = 2.0/N * np.abs(yf[0:N//2])

    return xf, vals
for label in ['no', 'yes', 'cat']:    

    wav = samples[label][0]

    rate, row =  wavfile.read(train_audio_path + label + '/' + wav)

    f, axes = plt.subplots(1, 2, figsize=(10, 3))

    axes[0].plot(range(len(row)), row, label=label)

    axes[0].legend()

    xf, vals = custom_fft(row, rate)

    axes[1].plot(xf, vals, label = label)

    axes[1].legend()

    plt.show()
for label in ['no', 'yes', 'cat']:    

    N = len(samples[label])

    valprom = None

    for i in range(N):

        wav = samples[label][i]

        rate, row =  wavfile.read(train_audio_path + label + '/' + wav)

        if rate != 16000 or len(row) != 16000:            

            continue

        xf, vals = custom_fft(row, rate)

        if i == 0:

            valprom = vals

        else:

            valprom = valprom * (1.0 * i / (i + 1)) + vals * (1.0 / i + 1)

    f, axes = plt.subplots(1, 1, figsize=(10, 3))        

    axes.plot(xf, valprom, label=label)

    axes.legend()

    plt.show()
import pywt

pywt.dwt([1, 2, 3, 4, 5, 6], 'db1')
from sklearn.mixture import GaussianMixture

gauss = GaussianMixture(n_components=10)

from sklearn.cluster import KMeans

kmean = KMeans(n_clusters=10)
for label in labels:    

    wav = samples[label][0]

    rate, row =  wavfile.read(train_audio_path + label + '/' + wav)    

    f, axes = plt.subplots(1, 2, figsize=(10, 3))

    axes[0].plot(range(len(row)), row, label=label)

    axes[0].legend()        

    cA, cD = pywt.dwt(row, 'db1')

    kmean.fit(np.array(list(zip(cA, cD))))

    axes[1].scatter(cA, cD, label = label)

    axes[1].scatter(*zip(*kmean.cluster_centers_))

    axes[1].legend()

    plt.show()
import time

f, axes = plt.subplots(1, 1, figsize=(10, 3))        

for label in ['no', 'yes', 'cat']:    

    for i in range(3):

        wav = samples[label][i]

        rate, row =  wavfile.read(train_audio_path + label + '/' + wav)

        if rate != 16000 or len(row) != 16000:            

            continue

        cA, cD = pywt.dwt(row, 'db1')        

        if i == 0:

            axes.scatter(cA, cD, alpha=0.1, label=label)

        else:

            axes.scatter(cA, cD, alpha=0.1)

axes.legend()

plt.show()
import time

Xraw = []

y = []

for label in infosamples.label:    

    start = time.time()

    if label == '_background_noise_':

        continue

    else:

        N = len(samples[label])       

        Nsampl = 0

        for i in range(N):

            wav = samples[label][i]

            rate, row =  wavfile.read(train_audio_path + label + '/' + wav)

            if rate != 16000 or len(row) != 16000:            

                continue

            xf, vals = custom_fft(row, rate)

            Xraw.append(vals)

            y.append(label)

            Nsampl += 1

    print('Label {} took {:.2f} seconds to extract {}/{} samples'.format(label, time.time() - start, Nsampl, N))
from sklearn.utils import shuffle

Xrnd = shuffle(Xraw)[:20000]
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

scaler = StandardScaler()

pca = PCA(n_components=100)

scaler.fit(Xrnd)

pca.fit(scaler.transform(Xrnd))
X = pca.transform(scaler.transform(Xraw))
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
from sklearn.preprocessing import FunctionTransformer

def select_columns(X, k):

    return X[:, :k]

trans = FunctionTransformer(func=select_columns, kw_args={'k': 36})
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

pipe = Pipeline([('trans', trans), ('clf', RandomForestClassifier())])

pipe.fit(Xtrain, ytrain)

pipe.score(Xtest, ytest)
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

pipe = Pipeline([('trans', trans), ('clf',  MLPClassifier(hidden_layer_sizes = (5, )))])

pipe.fit(Xtrain, ytrain)

pipe.score(Xtest, ytest)
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

pipe = Pipeline([('trans', trans), ('clf',  MLPClassifier(hidden_layer_sizes = (10, 5)))])

pipe.fit(Xtrain, ytrain)

pipe.score(Xtest, ytest)
from random import randint

from sklearn.utils import shuffle

def balance(X, y, label, N, bal=0.5):

    I1 = [i for i, yy in enumerate(y) if yy == label]

    N1 = len(I1)

    I0 = [i for i, yy in enumerate(y) if yy != label]

    N0 = len(I0)

    n1 = int(bal * N)

    n0 = N - n1

    Xb = np.array([X[I1[randint(0, N1 - 1)]] for _ in range(n1)] 

                   + [X[I0[randint(0, N1 - 1)]] for _ in range(n0)])

    yb = np.array([1 for _ in range(n1)] + [0 for _ in range(n0)])

    return shuffle(Xb, yb)    
from scipy.optimize import minimize

def linearbound(X, y):

    X0 = np.array([xx for xx, yy in zip(X, y) if yy == 0])

    X1 = np.array([xx for xx, yy in zip(X, y) if yy == 1])

    m0 = X0.mean(axis=0)    

    m1 = X1.mean(axis=0)

    s0 = np.cov(np.transpose(X0))

    s1 = np.cov(np.transpose(X1))

    def fun(a):

        delta = np.dot(a, (m0 - m1))

        s0a = np.dot(np.dot(a, s0), a) ** 0.5

        s1a = np.dot(np.dot(a, s1), a) ** 0.5         

        return 0.5 / (1.0 + (delta / (s0a + s1a))**2)

    res = minimize(fun, np.ones(len(m0)))

    return res.fun
from multiprocessing import Pool



N = 2000

def fun(label):

    start = time.time()

    Xb, yb = balance(X, y, label, N)

    err = linearbound(Xb, yb)

    elapsed = time.time() - start

    print('{}: {:.3%} (elapsed {:.3f})'.format(label, err, elapsed))

    

with Pool() as p:    

    p.map(fun, labels)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.preprocessing import FunctionTransformer



from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier



def select_columns(X, k):

    return X[:, :k]



def multi_train_clf(X, y):            

    trans = FunctionTransformer(func=select_columns)

    svc = SVC(kernel='poly', degree=2)

    mlp = MLPClassifier()

    pipe = Pipeline(steps=[('trans', trans), ('clf', svc)])

    grid_svc = dict(clf=[svc],

                    trans__kw_args=[{'k': k} for k in [35, 40, 45]],

                    clf__kernel=['poly', 'rbf'],

                    clf__degree=[2, 3, 4],                    

                    clf__C=[0.12, 0.2, 0.3], 

                    )   

    grid_mlp = dict(clf=[mlp],

                    trans__kw_args=[{'k': k} for k in [40, 45, 50]],

                    clf__hidden_layer_sizes = [(5, ), (10, ), (2, 4), (5, 1)],

                    clf__activation=['relu'],

                    )

    clf = GridSearchCV(pipe, [grid_svc, grid_mlp], n_jobs=-1, verbose=2, error_score=0)

    clf.fit(X, y)

    return clf



def train_clf(X, y):

    trans = FunctionTransformer(func=select_columns, kw_args={'k':35})

    svc = SVC(kernel='poly', degree=2, C=0.12)

    clf = Pipeline(steps=[('trans', trans), ('svc', svc)])

    clf.fit(X, y)

    return clf
from multiprocessing import Pool

from sklearn.model_selection import cross_val_score



def fun(label):

    Xb ,yb = balance(X, y, label, 2000)

    start = time.time()

    clf = train_clf(Xb, yb)

    elapsed = time.time() - start

    print('{}: {:.2%}. (elapsed time: {:.3f})'.format(label, cross_val_score(clf, Xb, yb).mean(), elapsed))

    

with Pool() as p:    

    p.map(fun, labels)
from multiprocessing import Pool, Manager



clfs = Manager().dict()

def fun(label):

    Xb ,yb = balance(Xtrain, ytrain, label, 3000, bal=0.2)

    Xbt, ybt = balance(Xtest, ytest, label, 2000, bal=1.0)

    start = time.time()

    clfs[label] = train_clf(Xb, yb)

    elapsed = time.time() - start

    print('{}: {:.2%}. (elapsed time: {:.3f})'.format(label, clfs[label].score(Xbt, ybt), elapsed))       

    

with Pool() as p:    

    p.map(fun, labels)    
from multiprocessing import Pool, Manager

from itertools import repeat



def fun(args):

    label, X, ys = args

    start = time.time()

    ys[label] = clfs[label].predict(X)

    elapsed = time.time() - start

    print('{} predicted (elapsed {:.3f})'.format(label, elapsed))



def semi_predict(X):

    ys = Manager().dict()

    with Pool() as p:

        p.map(fun, zip(labels, repeat(X), repeat(ys)))

    return np.stack([ys[label] for label in labels], axis=-1)    
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(semi_predict(Xtrain), ytrain)
ytestsup = tree.predict(semi_predict(Xtest))
sum([a == b for a, b in zip(ytestsup, ytest)]) / len(ytest)
# import random



# perm = np.array(range(1, 31))

# dctlabels = {label: i for i, label in enumerate(labels)}



# ynumb = np.array([dctlabels[label] for label in ytrain])



# def fun(yarr, ynumb, ytrain, perm):

#     cnt = 0

#     for row, rs, ro in zip(yarr, ynumb, ytrain):

#         if np.argmax([a * b for a, b in zip(row, perm)]) == rs:

#             cnt += 1

#     return 1.0 * cnt/ len(ynumb)



# ass = fun(yarr, ynumb, ytrain, perm)

# for _ in range(100):

#     i = random.randint(0, 29)

#     j = random.randint(0, 29)

#     perm[i], perm[j] = perm[j], perm[i]

#     newass = fun(yarr, ynumb, ytrain, perm)    

#     if newass > ass:

#         ass = newass

#         print(newass)

#     else:

#         perm[i], perm[j] = perm[j], perm[i]
import random

def sample(X, n):

    N = len(X)

    return np.array([X[random.randint(0, N - 1)] for i in range(n)])
import time



Xraw = []

yraw = []



def fun(label):

    start = time.time()

    N = len(samples[label])       

    for i in range(N):

        wav = samples[label][i]

        rate, row =  wavfile.read(train_audio_path + label + '/' + wav)

        if rate != 16000 or len(row) != 16000:            

            continue

        Xraw.append(row)

        yraw.append(label)

    print('Tiempo de lectura {}: {:.3f}'.format(label, time.time() - start))    



for label in labels:

    fun(label)
import time

import pywt

from sklearn.cluster import KMeans

from multiprocessing import Pool, Manager



points = Manager().dict()



def fun(label):

    start = time.time()

    N = len(samples[label])       

    Xpts = []

    for i in range(N):

        wav = samples[label][i]

        rate, row =  wavfile.read(train_audio_path + label + '/' + wav)

        if rate != 16000 or len(row) != 16000:            

            continue

        cA, cD = pywt.dwt(row, 'db1')   

        Xpts.extend(sample(list(zip(cA, cD)), 100))

    print('Tiempo de lectura {}: {:.3f}'.format(label, time.time() - start))    

    start = time.time()

    kmean = KMeans(n_clusters=40, n_jobs=1)    

    kmean.fit(sample(Xpts, 10000))

    points[label] = kmean.cluster_centers_

    print('Tiempo de clustering {}: {:.3f}'.format(label, time.time() - start))





with Pool() as p:

    p.map(fun, labels)

# for label in labels:

#     fun(label)
Xred = []

yred = []

for label in labels:

    n = len(points[label])

    Xred.extend([x for x in points[label]])

    yred.extend([label] * n)

Xred = np.array(Xred)

yred = np.array(yred)
yred
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(Xred, yred)
clf.score(Xred, yred)
idx = 20

cA, cD = pywt.dwt(Xraw[idx], 'db1')

data = list(zip(cA, cD))

kmean = KMeans(n_clusters=100, n_jobs=-1)    

kmean.fit(data)

data = kmean.cluster_centers_

dist, ind = clf.kneighbors(data)
Counter([yred[i] for i in ind.ravel()])
yraw[idx]