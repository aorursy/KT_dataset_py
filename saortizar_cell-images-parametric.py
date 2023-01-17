%load_ext line_profiler
import time

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import statistics as st



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
f, ax = plt.subplots(4, 2, figsize=(5, 10))

for k in range(4):

    for y in range(2):

        ax[k][y].imshow(sample(y, k))
N_samples = 2000

samples_uninfected = [sample(0, i) for i in range(N_samples)]

samples_infected = [sample(1, i) for i in range(N_samples)]
def X1(image):

    intensities = 1.0 - ((image[:,:,0] * image[:,:,0] \

                + image[:,:,1] * image[:,:,1] \

                + image[:,:,2] * image[:,:,2])) ** 0.5

    f_intensities = intensities[intensities < 0.99]

    min_int = np.min(f_intensities)

    max_int = np.max(f_intensities)

    assert max_int < 0.9

    return max_int - min_int

#%lprun -f X1 X1(samples_uninfected[0])

#start = time.time()

#X1(samples_uninfected[0])

#print(time.time() - start)
pool = Pool(4)

X_uninfected = pool.map(X1, samples_uninfected)

X_infected = pool.map(X1, samples_infected)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.hist(X_uninfected, color='b', density=True, alpha=0.4, label='uninfected')

ax.hist(X_infected, color='y', density=True, alpha=0.4, label='infected')

plt.legend()

plt.show()
def X2(image):

    intensities = 1.0 - (((image[:,:,0] * image[:,:,0] \

                + image[:,:,1] * image[:,:,1] \

                + image[:,:,2] * image[:,:,2])) / 3)

    f_intensities = intensities[intensities < 0.99]

    min_int = np.min(f_intensities)

    max_int = np.max(f_intensities)

    assert max_int < 0.99

    return max_int - min_int
N_samples = 2000

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
N_samples = 1000

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

X1_uninfected = pool.map(X1, samples_uninfected)

X1_infected = pool.map(X1, samples_infected)

X2_uninfected = pool.map(X2, samples_uninfected)

X2_infected = pool.map(X2, samples_infected)

X_uninfected = np.array(list(zip(X1_uninfected, X2_uninfected)))

X_infected = np.array(list(zip(X1_infected, X2_infected)))
mean_uninfected = np.median(X_uninfected, axis=0)

mean_infected = np.median(X_infected, axis=0)

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

plot(0.0, 1.1, -0.1, 0.6, mean_uninfected, cov_uninfected, ax)

plot(0.0, 1.1, -0.1, 0.6, mean_infected, cov_infected, ax)

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

#pool = Pool(4)

X1_uninfected_test = map(X1, samples_uninfected_test)

X1_infected_test = map(X1, samples_infected_test)

X2_uninfected_test = map(X2, samples_uninfected_test)

X2_infected_test = map(X2, samples_infected_test)

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

#fn1s = []

fns = []

#fp1s = []

fps = []

#fpr1s = []

fprs = []

#tpr1s = []

tprs = []

for r in costos:

    #phi1 = clasificador(r, p1, pdfX1_uninfected, pdfX1_infected)

    phi = clasificador(r, p1, pdfX_uninfected, pdfX_infected)

    #fp1, fn1 = fp_fn(phi1, p1, X1_uninfected_test, X1_infected_test)

    fp, fn = fp_fn(phi, p1, X_uninfected_test, X_infected_test)

    #fpr1, tpr1 = fpr_tpr(phi1, X1_uninfected_test, X1_infected_test)

    fpr, tpr = fpr_tpr(phi, X_uninfected_test, X_infected_test)

    #fn1s.append(fn1)

    fns.append(fn)

    #fp1s.append(fp1)

    fps.append(fp)

    #fpr1s.append(fpr1)

    fprs.append(fpr)

    #tpr1s.append(tpr1)

    tprs.append(tpr)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

#ax1.plot(costos, fn1s, label='One feature')

ax1.plot(costos, fns, label='Both features')

ax1.set_xlabel('Relative cost')

ax1.set_ylabel('False negatives')

ax1.grid()

ax1.legend()

#ax2.plot(costos, fp1s, label='One feature')

ax2.plot(costos, fps, label='Both features')

ax2.set_xlabel('Relative cost')

ax2.set_ylabel('False positives')

ax2.grid()

ax2.legend()

plt.show()
f, ax = plt.subplots(1, 1, figsize=(12, 3))

#ax.plot(fpr1s, tpr1s, label='One feature')

ax.plot(fprs, tprs, label='Both features')

ax.set_xlim(0.0, 1.0)

ax.set_ylim(0.0, 1.0)

ax.set_xlabel('False positive rate')

ax.set_ylabel('True positive rate')

ax.grid()

ax.legend()

plt.show()