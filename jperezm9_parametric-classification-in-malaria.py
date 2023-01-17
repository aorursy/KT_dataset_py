import matplotlib

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.cluster import KMeans

from multiprocessing import Pool

import cv2

import struct

import scipy

import scipy.misc

import scipy.cluster

import scipy.stats as st

import statsmodels as sm

from scipy.special import gamma as gammaf

from scipy.optimize import fmin

from scipy.stats import genextreme as gev

import numpy as np 

import pandas as pd

import os
root = "../input/cell_images/cell_images"

root_uninfected = root + '/Uninfected'

root_infected = root + '/Parasitized'

print(os.listdir(root))

files_uninfected = os.listdir(root_uninfected)

files_infected = os.listdir(root_infected)

print('{} uninfected files'.format(len(files_uninfected)))

print('{} infected files'.format(len(files_infected)))
# Extraction function

def sample(y, k):

    if y == 0:

        return mpimg.imread(os.path.join(root_uninfected, files_uninfected[k]))

    elif y == 1:

        return mpimg.imread(os.path.join(root_infected, files_infected[k]))

    else:

        raise ValueError

        

# Figure of 8 sub-samples

f, ax = plt.subplots(4, 2, figsize=(5, 10))

for k in range(4):

    for y in range(2):

        ax[k][y].imshow(sample(y, k))
# Definition feature extraction

def inf(im):

    

    # Clustering

    NUM_CLUSTERS = 7

    ar = np.asarray(im)

    shape = ar.shape

    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         

    counts, bins = scipy.histogram(vecs, len(codes))    

    index_max = scipy.argmax(counts)                  

    peak = codes[index_max]

    colour = ''.join(chr(int(c)) for c in peak).encode()

    index_min = scipy.argmin(counts)                   

    peak2 = codes[index_min]

    colour2 = ''.join(chr(int(c)) for c in peak2).encode()

    thresholded=cv2.inRange(im,(0,0,0),(0,0,0))

    

    # Transform clustered image to black and white scale

    im=im+cv2.cvtColor(thresholded,cv2.COLOR_GRAY2BGR)

    th=cv2.inRange(im,(0,0,0),(peak2[0],peak2[1],peak2[2]))

    im=cv2.cvtColor(th,cv2.COLOR_GRAY2BGR)

    

    # Count the proportion of black and white pixels inside the image

    I, J, K = im.shape

    I, J, K

    bla = 0

    neg = 0

    for i in range(I):

        for j in range(J):

            if all(np.isclose(im[i, j], [0, 0, 0])):

                neg += 1

            else:

                bla += 1

    tot = neg+bla

    Pneg = neg/tot

    Pbla = bla/tot

    return Pneg 
N_samples = 1000

samples_uninfected = [sample(0, i) for i in range(N_samples)]

samples_infected = [sample(1, i) for i in range(N_samples)]
pool = Pool(8)

X_uninfected = pool.map(inf, samples_uninfected)

X_infected = pool.map(inf, samples_infected)
plt.subplot(2,1,1)

plt.hist(X_uninfected, bins=60, color='b', density=True, alpha=0.4, label='uninfected')

plt.legend()

plt.title('X uninfected')



plt.subplot(2,1,2)

plt.hist(X_infected, bins=60, color='y', density=True, alpha=0.4, label='infected')

plt.legend()

plt.title('X infected')



plt.subplots_adjust(top=2, bottom=0.08, left=0.10, right=2, hspace=0.25, wspace=0.35)

plt.show()
b = np.asarray(X_uninfected)<=0.7

X_uninfected = np.extract(b, np.asarray(X_uninfected))

plt.subplot(2,1,1)

plt.hist(X_uninfected, bins=60, color='b', density=True, alpha=0.4, label='uninfected')

plt.legend()

plt.title('X uninfected')



c = np.asarray(X_infected)>0.9

X_infected = np.extract(c, np.asarray(X_infected))

plt.subplot(2,1,2)

plt.hist(X_infected, bins=60, color='y', density=True, alpha=0.4, label='infected')

plt.legend()

plt.title('X infected')



plt.subplots_adjust(top=2, bottom=0.08, left=0.10, right=2, hspace=0.25, wspace=0.35)

plt.show()

# Definition of the probability function

def lognormal(mu, sigma):

    def fun(x):

        return np.exp(- (np.log(x) - mu) ** 2 /(2 * sigma ** 2)) / (x * sigma * (2 * np.pi) ** 0.5) 

    return fun



# Calculation of the probability function and histogram

log_Xuninfected = np.log(X_uninfected)

mu0 = np.mean(log_Xuninfected)

sigma0 = np.std(log_Xuninfected)

print('mu: {:.3f}, sigma: {:.3f}'.format(mu0, sigma0))



X0 = lognormal(mu0, sigma0)

plt.figure(figsize=(17,5))

dom = np.linspace(0, 1, 100)

plt.hist(X_uninfected, bins=60, color='b', density=True, alpha=0.4, label='uninfected')

plt.plot(dom, list(map(X0, dom)), label='uninfected fit')

plt.legend()

plt.title('X uninfected')

plt.show()
# Definition of the probability function

data=X_infected

(w1, w2, w3, w4)=st.exponweib.fit(data,floc=0, f0=np.median(data))

def weib(n,a):

    def fun(x):

        return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

    return fun



# Calculation of the probability function and histogram

X1 = weib(w1, w2)

plt.figure(figsize=(17,5))

dom = np.linspace(min(data),max(data),len(data))

plt.hist(X_infected, bins=60, color='y', density=True, alpha=0.4, label='infected')

plt.plot(dom, list(map(X1, dom)), label='infected fit')

plt.legend()

plt.title('X infected')

plt.show()
# Choice of random samples

N_samples_test = 1000

samples_uninfected_test = [sample(0, 1000 + i) for i in range(N_samples_test)]

samples_infected_test = [sample(1, 1000 + i) for i in range(N_samples_test)]

pool = Pool(4)



# Feature extraction 

X_uninfected_test = pool.map(inf, samples_uninfected_test)

X_infected_test = pool.map(inf, samples_infected_test)
# Classifier definition

def clasificador(r, p1, X0, X1):

    def phi(x):

        q0 = r * p1 * X0(x)

        q1 = (1 - p1) * X1(x)

        if q0 < q1:

            return 0

        else:

            return 1

    return phi
costos = [1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 

          1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 

          1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30, 1e31, 1e32, 1e33, 1e34, 1e35,

         1e36, 1e37, 1e38, 1e39, 1e40, 1e41, 1e42, 1e43, 1e44, 1e45, 1e46, 1e47, 1e48, 1e49, 1e50,

         1e51, 1e52, 1e53, 1e54, 1e55, 1e56, 1e57, 1e58, 1e59, 1e60,

         1e61, 1e62, 1e63, 1e64, 1e65, 1e66, 1e67, 1e68, 1e69, 1e70,

         1e71, 1e72, 1e73, 1e74, 1e75, 1e76, 1e77, 1e78, 1e79, 1e80,

         1e81, 1e82, 1e83, 1e84, 1e85, 1e86, 1e87, 1e88, 1e89, 1e90] 

p1 = 0.5 / 10000
# False positive rate

def fp_fn(phi, p1, X_uninfected, X_infected):

    fpr = sum(map(phi, X_uninfected)) / len(X_uninfected)

    fp = fpr * (1 - p1)

    fnr = 1.0 - sum(map(phi, X_infected)) / len(X_infected)

    fn = fnr * p1

    return fp, fn



# True positive rate

def fpr_tpr(phi, X_uninfected, X_infected):

    fpr = sum(map(phi, X_uninfected)) / len(X_uninfected)

    tpr = sum(map(phi, X_infected)) / len(X_infected)

    return fpr, tpr
fn1s = []

fp1s = []

fpr1s = []

tpr1s = []

for r in costos:

    phi1 = clasificador(r, p1, X0, X1)

    fp1, fn1 = fp_fn(phi1, p1, X_uninfected_test, X_infected_test)

    fpr1, tpr1 = fpr_tpr(phi1, X_uninfected_test, X_infected_test)

    fn1s.append(fn1)

    fp1s.append(fp1)

    fpr1s.append(fpr1)

    tpr1s.append(tpr1)

    

plt.figure(figsize=(17,7))

plt.scatter(tpr1s, fpr1s)

plt.plot(tpr1s, fpr1s)

plt.xlabel('True Positive Rate')

plt.ylabel('False Positive Rate')

plt.title('ROC Curve')

plt.grid()

plt.show()
