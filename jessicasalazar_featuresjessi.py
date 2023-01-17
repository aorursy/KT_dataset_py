%load_ext line_profiler
import time

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from IPython.display import Image

from multiprocessing import Pool

from scipy.stats import multivariate_normal
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

root = "../input/cell_images/cell_images"

root_uninfected = root + '/Uninfected'

root_infected = root + '/Parasitized'

print(os.listdir(root))



# Any results you write to the current directory are saved as output.
files_uninfected = os.listdir(root_uninfected)

files_infected = os.listdir(root_infected)
print('{} uninfected files'.format(len(files_uninfected)))

print('{} infected files'.format(len(files_infected)))
def sample(y, k):

    if y == 0:

        return mpimg.imread(os.path.join(root_uninfected, files_uninfected[k]))

    elif y == 1:

        return mpimg.imread(os.path.join(root_infected, files_infected[k]))

    else:

        raise ValueError
sample(0, 0).shape
f, ax = plt.subplots(4, 2, figsize=(10, 20))

for k in range(4):

    for y in range(2):

        ax[k][y].imshow(sample(y, k))
def X1_0(image):

    I, J, K = image.shape

    I, J, K

    colores = []

    for i in range(I):

        for j in range(J):

            if all(np.isclose(image[i, j], [0, 0, 0])):

                pass

            else:

                colores.append(image[i, j])

    intensidades = [1 - np.linalg.norm(x / 3.0) for x in colores]

    min_int = min(intensidades)

    max_int = max(intensidades)

    assert max_int <= 0.9

    return max_int - min_int
N_samples = 50

samples_uninfected = [sample(0, i) for i in range(N_samples)]

samples_infected = [sample(1, i) for i in range(N_samples)]
pool = Pool(4)

X_uninfected = pool.map(X1_0, samples_uninfected)

X_infected = pool.map(X1_0, samples_infected)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.hist(X_uninfected, color='b', density=True, alpha=0.4, label='uninfected')

ax.hist(X_infected, color='y', density=True, alpha=0.4, label='infected')

plt.legend()

plt.show()
%lprun -f X1_0 X1_0(samples_uninfected[0])
def X1_1(image):

    I, J, K = image.shape

    I, J, K

    colores = []

    for i in range(I):

        for j in range(J):

            if all(np.isclose(image[i, j], [0, 0, 0])):

                pass

            else:

                colores.append(image[i, j])

    intensidades = [1 - np.linalg.norm(x / 3.0) for x in colores]

    min_int = min(intensidades)

    max_int = max(intensidades)

    assert max_int <= 0.9

    return max_int - min_int

%lprun -f X1_1 X1_1(samples_uninfected[0])
def intensity(x):

    return 1 - np.linalg.norm(x / 3.0)



def X1_2(image):

    intensities = np.apply_along_axis(intensity, 2, image)

    f_intensities = intensities[intensities < 0.99]    

    min_int = np.min(f_intensities)

    max_int = np.max(f_intensities)

    assert max_int <= 0.9

    return max_int - min_int



%lprun -f X1_2 X1_2(samples_uninfected[0])
def X1_3(image):

    intensities = 1.0 - ((image[:,:,0] * image[:,:,0] \

                + image[:,:,1] * image[:,:,1] \

                + image[:,:,2] * image[:,:,2]) / 3.0) ** 0.5

    f_intensities = intensities[intensities < 0.99]

    min_int = np.min(f_intensities)

    max_int = np.max(f_intensities)

    assert max_int < 0.9

    return max_int - min_int

%lprun -f X1_3 X1_3(samples_uninfected[0])

start = time.time()

X1_3(samples_uninfected[0])

print(time.time() - start)
import theano

import theano.tensor as T

A = T.tensor3('A')

it = 1 - (A[:,:,0] ** 2 + A[:,:,1] ** 2 + A[:,:,2] ** 2) ** 0.5

s = T.max(T.switch(T.lt(it, 0.99), it, 0.0)) - T.min(it) 

X1_4 = theano.function([A], s)

start = time.time()

X1_4(samples_uninfected[0])

print(time.time() - start)
X1 = X1_3
def X2(image):

    intensities = 1.0 - ((image[:,:,0] * image[:,:,0] \

            + image[:,:,1] * image[:,:,1] \

            + image[:,:,2] * image[:,:,2]) / 3.0) ** 0.5

    f_intensities = intensities[intensities < 0.99]

    max_int = np.max(f_intensities)

    assert max_int < 0.9

    N = f_intensities.size

    max_intensities = f_intensities[f_intensities > max_int * (1 - 0.05)]

    n = max_intensities.size

    return np.log((n + 1) / N)
N_samples = 500

samples_uninfected = [sample(0, i) for i in range(N_samples)]

samples_infected = [sample(1, i) for i in range(N_samples)]
pool = Pool(4)

X2_uninfected = pool.map(X2, samples_uninfected)

X2_infected = pool.map(X2, samples_infected)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.hist(X2_uninfected, bins=20, color='b', density=True, alpha=0.4, label='uninfected')

ax.hist(X2_infected, bins=20, color='y', density=True, alpha=0.4, label='infected')

plt.legend()

plt.show()
N_samples = 500

samples_uninfected = [sample(0, i) for i in range(N_samples)]

samples_infected = [sample(1, i) for i in range(N_samples)]
pool = Pool(4)

X1_uninfected = pool.map(X1, samples_uninfected)

X1_infected = pool.map(X1, samples_infected)

X2_uninfected = pool.map(X2, samples_uninfected)

X2_infected = pool.map(X2, samples_infected)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.scatter(X1_uninfected, X2_uninfected, color='b', alpha=0.4, label='uninfected')

ax.scatter(X1_infected, X2_infected, color='r', alpha=0.4, label='infected')

plt.legend()

plt.show()
N_samples = 1000

samples_uninfected = [sample(0, i) for i in range(N_samples)]

samples_infected = [sample(1, i) for i in range(N_samples)]

pool = Pool(4)

X_uninfected = pool.map(X1, samples_uninfected)

X_infected = pool.map(X1, samples_infected)
log_Xuninfected = np.log(X_uninfected)

mu0 = np.mean(log_Xuninfected)

sigma0 = np.std(log_Xuninfected)

print('mu: {:.3f}, sigma: {:.3f}'.format(mu0, sigma0))
mu1 = np.mean(X_infected)

sigma1 = np.std(X_infected)

print('mu: {:.3f}, sigma: {:.3f}'.format(mu1, sigma1))
def lognormal(mu, sigma):

    def fun(x):

        return np.exp(- (np.log(x) - mu) ** 2 /(2 * sigma ** 2)) / (x * sigma * (2 * np.pi) ** 0.5) 

    return fun
def norm(mu, sigma):

    def fun(x):

        return np.exp(-0.5 * ((x - mu)/ sigma)** 2)/((2 * np.pi) ** 0.5 * sigma)

    return fun
pdfX1_uninfected = lognormal(mu0, sigma0)

pdfX1_infected = norm(mu1, sigma1)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

dom = np.linspace(0, 1, 100)

ax.hist(X_uninfected, bins=30, color='b', density=True, alpha=0.4, label='uninfected')

ax.hist(X_infected, bins=30, color='y', density=True, alpha=0.4, label='infected')

ax.plot(dom, list(map(pdfX1_uninfected, dom)), label='uninfected')

ax.plot(dom, list(map(pdfX1_infected, dom)), label='infected')

plt.legend()

plt.show()
sample(0,0).shape
def caract_1(image):

    intensities = 1.0 - ((image[:,:,0] * image[:,:,0] \

                + image[:,:,1] * image[:,:,1] \

                + image[:,:,2] * image[:,:,2]) / 3.0) ** 0.5

    f_intensities = intensities[intensities < 0.99]

    return np.std(f_intensities)

%lprun -f caract_1 caract_1(samples_uninfected[0])
pool = Pool(4)

caract1_uninfected = pool.map(caract_1, samples_uninfected)

caract1_infected = pool.map(caract_1, samples_infected)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.hist(caract1_uninfected, bins=20, color='b', density=True, alpha=0.4, label='uninfected')

ax.hist(caract1_infected, bins=20, color='y', density=True, alpha=0.4, label='infected')

plt.legend()

plt.show()
def caract_2(image):

    intensities = 1.0 - ((image[:,:,0] * image[:,:,0] \

                + image[:,:,1] * image[:,:,1] \

                + image[:,:,2] * image[:,:,2]) / 3.0) ** 0.5

    f_intensities = intensities[intensities < 0.99]

    min_int = np.min(f_intensities)

    max_int = np.max(f_intensities)

    f1_intensities = 1.0*(f_intensities-min_int)/(max_int-min_int)

    return np.mean(f1_intensities)

%lprun -f caract_2 caract_2(samples_uninfected[0])
pool = Pool(4)

caract2_uninfected = pool.map(caract_2, samples_uninfected)

caract2_infected = pool.map(caract_2, samples_infected)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.hist(caract2_uninfected, bins=20, color='g', density=True, alpha=0.4, label='uninfected')

ax.hist(caract2_infected, bins=20, color='r', density=True, alpha=0.4, label='infected')

plt.legend()

plt.show()
X1 = caract_2

X2 = caract_1
N_samples = 2000

samples_uninfected = [sample(0, i) for i in range(N_samples)]

samples_infected = [sample(1, i) for i in range(N_samples)]

pool = Pool(4)

X1_uninfected = pool.map(X1, samples_uninfected)

X1_infected = pool.map(X1, samples_infected)

X2_uninfected = pool.map(X2, samples_uninfected)

X2_infected = pool.map(X2, samples_infected)

X_uninfected = np.array(list(zip(X1_uninfected, X2_uninfected)))

X_infected = np.array(list(zip(X1_infected, X2_infected)))
mean_uninfected = np.mean(X_uninfected, axis=0)

mean_infected = np.mean(X_infected, axis=0)

print('Media de no infectados: {}'.format(mean_uninfected))

print('Media de infectados: {}'.format(mean_infected))
cov_uninfected = np.cov(X_uninfected.T)

cov_infected = np.cov(X_infected.T)

print('Covarianza de no infectados: \n{}'.format(cov_uninfected))

print('Covarianza de infectados: \n{}'.format(cov_infected))
pdfX_uninfected = multivariate_normal(mean_uninfected, cov_uninfected).pdf

pdfX_infected = multivariate_normal(mean_infected, cov_infected).pdf
def plot(x0, x1, y0, y1, mean, cov, ax):

    x, y = np.mgrid[x0:x1:.01, y0:y1:.01]

    pos = np.dstack((x, y))

    rv = multivariate_normal(mean, cov)

    ax.contourf(x, y, rv.pdf(pos), alpha=0.2, levels=10)

    

f, ax = plt.subplots(1, 1, figsize=(12, 3))

plot(0.0, 1, 0, 0.15, mean_uninfected, cov_uninfected, ax)

plot(0.0, 1, 0, 0.15, mean_infected, cov_infected, ax)

ax.scatter(X1_uninfected, X2_uninfected, color='b', alpha=0.4, label='uninfected')

ax.scatter(X1_infected, X2_infected, color='r', alpha=0.4, label='infected')

plt.legend()

plt.show()
def clasificador(r, p1, pdfX_uninfected, pdfX_infected):

    def phi(x):

        q0 = r * p1 * pdfX_infected(x)

        q1 = (1 - p1) * pdfX_uninfected(x)

        if q0 < q1:

            return 0

        else:

            return 1

    return phi
p1 = 0.5 / 10000
costos = [1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
range_samples = range(10000, 11000) 

samples_uninfected_test = [sample(0, i) for i in range_samples]

samples_infected_test = [sample(1, i) for i in range_samples]

pool = Pool(4)

X1_uninfected_test = pool.map(X1, samples_uninfected_test)

X1_infected_test = pool.map(X1, samples_infected_test)

X2_uninfected_test = pool.map(X2, samples_uninfected_test)

X2_infected_test = pool.map(X2, samples_infected_test)

X_uninfected_test = list(zip(X1_uninfected_test, X2_uninfected_test))

X_infected_test = list(zip(X1_infected_test, X2_infected_test))


def fp_fn(phi, p1, X_uninfected, X_infected):

    fpr = sum(map(phi, X_uninfected)) / len(X_uninfected)

    fp = fpr * (1 - p1)

    fnr = 1.0 - sum(map(phi, X_infected)) / len(X_infected)

    fn = fnr * p1

    return fp, fn



def fpr_tpr(phi, X_uninfected, X_infected):

    fpr = sum(map(phi, X_uninfected)) / len(X_uninfected)

    tpr = sum(map(phi, X_infected)) / len(X_infected)

    return fpr, tpr

fn1s = []

fns = []

fp1s = []

fps = []

fpr1s = []

fprs = []

tpr1s = []

tprs = []

for r in costos:

    phi1 = clasificador(r, p1, pdfX1_uninfected, pdfX1_infected)

    phi = clasificador(r, p1, pdfX_uninfected, pdfX_infected)

    fp1, fn1 = fp_fn(phi1, p1, X1_uninfected_test, X1_infected_test)

    fp, fn = fp_fn(phi, p1, X_uninfected_test, X_infected_test)

    fpr1, tpr1 = fpr_tpr(phi1, X1_uninfected_test, X1_infected_test)

    fpr, tpr = fpr_tpr(phi, X_uninfected_test, X_infected_test)

    fn1s.append(fn1)

    fns.append(fn)

    fp1s.append(fp1)

    fps.append(fp)

    fpr1s.append(fpr1)

    fprs.append(fpr)

    tpr1s.append(tpr1)

    tprs.append(tpr)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

ax1.plot(costos, fn1s, label='One feature')

ax1.plot(costos, fns, label='Both features')

ax1.set_xlabel('Relative cost')

ax1.set_ylabel('False negatives')

ax1.grid()

ax1.legend()

ax2.plot(costos, fp1s, label='One feature')

ax2.plot(costos, fps, label='Both features')

ax2.set_xlabel('Relative cost')

ax2.set_ylabel('False positives')

ax2.grid()

ax2.legend()

plt.show()
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.plot(fpr1s, tpr1s, label='One feature')

ax.plot(fprs, tprs, label='Both features')

ax.set_xlim(0.0, 1.0)

ax.set_ylim(0.0, 1.0)

ax.set_xlabel('False positive rate')

ax.set_ylabel('True positive rate')

ax.grid()

ax.legend()

plt.show()