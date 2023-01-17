import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from IPython.display import Image

from multiprocessing import Pool
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

files_uninfected = [item for item in files_uninfected if item[-3:] != ".db"]

files_infected = [item for item in files_infected if item[-3:] != ".db"]
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
def X1(image):

    intensities = 1.0 - ((image[:,:,0] * image[:,:,0] \

                + image[:,:,1] * image[:,:,1] \

                + image[:,:,2] * image[:,:,2]) / 3.0) ** 0.5

    f_intensities = intensities[intensities < 0.99]

    min_int = np.min(f_intensities)

    max_int = np.max(f_intensities)

    assert max_int < 0.9

    return max_int - min_int
def X2(image):

    intensities = 1.0 - (image[:,:,0] * image[:,:,0] \

                + image[:,:,1] * image[:,:,1] \

                + image[:,:,2] * image[:,:,2]) ** 0.5

    f_intensities = intensities[intensities < 0.99]

    max_int = np.max(f_intensities)

    assert max_int < 0.9

    int_diff = max_int - f_intensities

    mean = np.mean(int_diff)

    return mean
N_samples = 2000

samples_uninfected = [sample(0, i) for i in range(N_samples)]

samples_infected = [sample(1, i) for i in range(N_samples)]

pool = Pool(4)

X2_uninfected = pool.map(X2, samples_uninfected)

X2_infected = pool.map(X2, samples_infected)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.hist(X2_uninfected, bins=30, color='b', density=True, alpha=0.4, label='uninfected')

ax.hist(X2_infected, bins=30, color='y', density=True, alpha=0.4, label='infected')

plt.legend()

plt.show()

logX2_uninfected = np.log(X2_uninfected)

mu0 = np.mean(logX2_uninfected)

sigma0 = np.std(logX2_uninfected)

print('mu: {:.3f}, sigma: {:.3f}'.format(mu0, sigma0))
mu1 = np.mean(X2_infected)

sigma1 = np.std(X2_infected)

print('mu: {:.3f}, sigma: {:.3f}'.format(mu1, sigma1))

def norm(mu, sigma):

    def fun(x):

        return np.exp(-0.5 * ((x - mu)/ sigma)** 2)/((2 * np.pi) ** 0.5 * sigma)

    return fun
def lognormal(mu, sigma):

    def fun(x):

        return np.exp(- (np.log(x) - mu) ** 2 /(2 * sigma ** 2)) / (x * sigma * (2 * np.pi) ** 0.5) 

    return fun
pdfX2_0 = lognormal(mu0, sigma0)

pdfX2_1 = norm(mu1, sigma1)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

dom = np.linspace(0.01, 1, 100)

ax.hist(X2_uninfected, bins=30, color='b', density=True, alpha=0.4, label='uninfected')

ax.hist(X2_infected, bins=30, color='y', density=True, alpha=0.4, label='infected')

ax.plot(dom, list(map(pdfX2_0, dom)), label='uninfected')

ax.plot(dom, list(map(pdfX2_1, dom)), label='infected')

plt.legend()

plt.show()
def clasificador(r, p1, pdf_uninfected, pdf_infected):

    def phi(x):

        q0 = r * p1 * pdf_infected(x) 

        q1 = (1 - p1) * pdf_uninfected(x)

        if q0 < q1:

            return 0

        else:

            return 1

    return phi
N_samples_test = 2000

pool = Pool(4)

samples_uninfected_test = [sample(0, 2000 + i) for i in range(N_samples_test)]

samples_infected_test = [sample(1, 2000 + i) for i in range(N_samples_test)]

X2_uninfected_test = pool.map(X2, samples_uninfected_test)

X2_infected_test = pool.map(X2, samples_infected_test)
phi = clasificador(1, 0.5, pdfX2_0, pdfX2_1)

print(sum(list(map(phi, X2_uninfected_test))))

print(sum(list(map(phi, X2_infected_test))))
p1 = 0.5 / 10000

N_samples_test = 1000

pool = Pool(4)

costos = range(1000, 100000, 1000)

p1 = 0.5 / 10000

f, ax = plt.subplots(1, 1, figsize=(13, 4))

fprs = []

tprs = []

for costo in costos:

    phi = clasificador(costo, p1, pdfX2_0, pdfX2_1)

    Y_uninfected_test = list(map(phi, X2_uninfected_test))

    Y_infected_test = list(map(phi, X2_infected_test))

    fpr = sum(Y_uninfected_test) / N_samples_test

    pnr = (len(Y_infected_test) - sum(Y_infected_test)) / N_samples_test

    tpr = 1 - pnr

    fprs.append(fpr)

    tprs.append(tpr)

    ax.scatter(fpr, tpr, color='b')

    ax.set_xlabel('true positive rate')

    ax.set_ylabel('positive negative rate')

    #ax.text(fp, np, '{}'.format(costo))

#fprs, tprs = list(zip(*sorted(list(zip(fprs, tprs)))))

plt.plot(fprs, tprs)

plt.grid()

plt.show()



pool = Pool(4)

X1_uninfected = pool.map(X1, samples_uninfected)

X1_infected = pool.map(X1, samples_infected)

f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.hist(X1_uninfected, bins=30, color='b', density=True, alpha=0.4, label='uninfected')

ax.hist(X1_infected, bins=30, color='y', density=True, alpha=0.4, label='infected')

plt.legend()

plt.show()
log_X1uninfected = np.log(X1_uninfected)

mu0 = np.mean(log_X1uninfected)

sigma0 = np.std(log_X1uninfected)

print('mu: {:.3f}, sigma: {:.3f}'.format(mu0, sigma0))
mu1 = np.mean(X1_infected)

sigma1 = np.std(X1_infected)

print('mu: {:.3f}, sigma: {:.3f}'.format(mu1, sigma1))

pdfX1_0 = lognormal(mu0, sigma0)

pdfX1_1 = norm(mu1, sigma1)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

dom = np.linspace(0.001, 1, 100)

ax.hist(X1_uninfected, bins=30, color='b', density=True, alpha=0.4, label='uninfected')

ax.hist(X1_infected, bins=30, color='y', density=True, alpha=0.4, label='infected')

ax.plot(dom, list(map(pdfX1_0, dom)), label='uninfected')

ax.plot(dom, list(map(pdfX1_1, dom)), label='infected')

plt.legend()

plt.show()
X1_uninfected_test = pool.map(X1, samples_uninfected_test)

X1_infected_test = pool.map(X1, samples_infected_test)

phi = clasificador(1, 0.5, pdfX1_0, pdfX1_1)
print(sum(list(map(phi, X1_uninfected_test))))

print(sum(list(map(phi, X1_infected_test))))
costos = range(1000, 100000, 1000)

p1 = 0.5 / 10000

f, ax = plt.subplots(1, 1, figsize=(13, 4))

fprs = []

tprs = []

fprs2 = []

tprs2 = []

for costo in costos:

    phi = clasificador(costo, p1, pdfX2_0, pdfX2_1)

    phi2 = clasificador(costo, p1, pdfX1_0, pdfX1_1)

    Y_uninfected_test = list(map(phi, X2_uninfected_test))

    Y_infected_test = list(map(phi, X2_infected_test))

    Y_uninfected_test2 = list(map(phi2, X1_uninfected_test))

    Y_infected_test2 = list(map(phi2, X1_infected_test))

    fpr = sum(Y_uninfected_test) / N_samples_test

    pnr = (len(Y_infected_test) - sum(Y_infected_test)) / N_samples_test

    tpr = 1 - pnr

    fpr2 = sum(Y_uninfected_test2) / N_samples_test

    pnr2 = (len(Y_infected_test2) - sum(Y_infected_test2)) / N_samples_test

    tpr2 = 1 - pnr2

    fprs.append(fpr)

    tprs.append(tpr)

    fprs2.append(fpr2)

    tprs2.append(tpr2)

    ax.scatter(fpr, tpr, color='b')

    ax.scatter(fpr2, tpr2, color='r')

    ax.set_xlabel('true positive rate')

    ax.set_ylabel('positive negative rate')

    #ax.text(fp, np, '{}'.format(costo))

#fprs, tprs = list(zip(*sorted(list(zip(fprs, tprs)))))

plt.plot(fprs, tprs)

plt.plot(fprs2, tprs2)

plt.grid()

plt.show()

def X3(image):

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
pool = Pool(4)

X3_uninfected = pool.map(X3, samples_uninfected)

X3_infected = pool.map(X3, samples_infected)
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.scatter(X1_uninfected, X3_uninfected, color='b', alpha=0.4, label='uninfected')

ax.scatter(X1_infected, X3_infected, color='r', alpha=0.4, label='infected')

plt.legend()

plt.show()
X_uninfected = np.array(list(zip(X1_uninfected, X3_uninfected)))

X_infected = np.array(list(zip(X1_infected, X3_infected)))

pool = Pool(4)

X3_uninfected_test = pool.map(X3, samples_uninfected_test)

X3_infected_test = pool.map(X3, samples_infected_test)

X_uninfected_test = np.array(list(zip(X1_uninfected_test, X3_uninfected_test)))

X_infected_test = np.array(list(zip(X1_infected_test, X3_infected_test)))
mean_uninfected = np.mean(X_uninfected, axis=0)

mean_infected = np.mean(X_infected, axis=0)

print('Media de no infectados: {}'.format(mean_uninfected))

print('Media de infectados: {}'.format(mean_infected))
cov_uninfected = np.cov(X_uninfected.T)

cov_infected = np.cov(X_infected.T)

print('Covarianza de no infectados: \n{}'.format(cov_uninfected))

print('Covarianza de infectados: \n{}'.format(cov_infected))
from scipy.stats import multivariate_normal



pdfX_0 = multivariate_normal(mean_uninfected, cov_uninfected).pdf

pdfX_1 = multivariate_normal(mean_infected, cov_infected).pdf



def plot(x0, x1, y0, y1, mean, cov, ax):

    x, y = np.mgrid[x0:x1:.01, y0:y1:.01]

    pos = np.dstack((x, y))

    rv = multivariate_normal(mean, cov)

    ax.contourf(x, y, rv.pdf(pos), alpha=0.2, levels=10)

    

f, ax = plt.subplots(1, 1, figsize=(12, 3))

plot(0.0, 0.6, -11, 0, mean_uninfected, cov_uninfected, ax)

plot(0.0, 0.6, -11, 0, mean_infected, cov_infected, ax)

ax.scatter(X1_uninfected, X3_uninfected, color='b', alpha=0.4, label='uninfected')

ax.scatter(X1_infected, X3_infected, color='r', alpha=0.4, label='infected')

plt.legend()

plt.show()
phi = clasificador(1, 0.5, pdfX_0, pdfX_1) 

print(sum(list(map(phi, X_uninfected_test))))

print(sum(list(map(phi, X_infected_test))))
costos = range(1000, 100000, 1000)

p1 = 0.5 / 10000

f, ax = plt.subplots(1, 1, figsize=(13, 4))

fprs = []

tprs = []

fprs2 = []

tprs2 = []

for costo in costos:

    phi = clasificador(costo, p1, pdfX2_0, pdfX2_1)

    phi2 = clasificador(costo, p1, pdfX_0, pdfX_1)

    Y_uninfected_test = list(map(phi, X2_uninfected_test))

    Y_infected_test = list(map(phi, X2_infected_test))

    Y_uninfected_test2 = list(map(phi2, X_uninfected_test))

    Y_infected_test2 = list(map(phi2, X_infected_test))

    fpr = sum(Y_uninfected_test) / N_samples_test

    pnr = (len(Y_infected_test) - sum(Y_infected_test)) / N_samples_test

    tpr = 1 - pnr

    fpr2 = sum(Y_uninfected_test2) / N_samples_test

    pnr2 = (len(Y_infected_test2) - sum(Y_infected_test2)) / N_samples_test

    tpr2 = 1 - pnr2

    fprs.append(fpr)

    tprs.append(tpr)

    fprs2.append(fpr2)

    tprs2.append(tpr2)

    ax.scatter(fpr, tpr, color='b')

    ax.scatter(fpr2, tpr2, color='r')

    ax.set_xlabel('true positive rate')

    ax.set_ylabel('positive negative rate')

    #ax.text(fp, np, '{}'.format(costo))

#fprs, tprs = list(zip(*sorted(list(zip(fprs, tprs)))))

plt.plot(fprs, tprs)

plt.plot(fprs2, tprs2)

plt.grid()

plt.show()
Xb_uninfected = np.array(list(zip(X2_uninfected, X3_uninfected)))

Xb_infected = np.array(list(zip(X2_infected, X3_infected)))

Xb_uninfected_test = np.array(list(zip(X2_uninfected_test, X3_uninfected_test)))

Xb_infected_test = np.array(list(zip(X2_infected_test, X3_infected_test)))
mean_uninfected = np.mean(Xb_uninfected, axis=0)

mean_infected = np.mean(Xb_infected, axis=0)

print('Media de no infectados: {}'.format(mean_uninfected))

print('Media de infectados: {}'.format(mean_infected))
cov_uninfected = np.cov(X_uninfected.T)

cov_infected = np.cov(X_infected.T)

print('Covarianza de no infectados: \n{}'.format(cov_uninfected))

print('Covarianza de infectados: \n{}'.format(cov_infected))
pdfXb_0 = multivariate_normal(mean_uninfected, cov_uninfected).pdf

pdfXb_1 = multivariate_normal(mean_infected, cov_infected).pdf



def plot(x0, x1, y0, y1, mean, cov, ax):

    x, y = np.mgrid[x0:x1:.01, y0:y1:.01]

    pos = np.dstack((x, y))

    rv = multivariate_normal(mean, cov)

    ax.contourf(x, y, rv.pdf(pos), alpha=0.2, levels=10)

    

f, ax = plt.subplots(1, 1, figsize=(12, 3))

plot(0.0, 0.6, -11, 0, mean_uninfected, cov_uninfected, ax)

plot(0.0, 0.6, -11, 0, mean_infected, cov_infected, ax)

ax.scatter(X2_uninfected, X3_uninfected, color='b', alpha=0.4, label='uninfected')

ax.scatter(X2_infected, X3_infected, color='r', alpha=0.4, label='infected')

plt.legend()

plt.show()
phi = clasificador(1, 0.5, pdfXb_0, pdfXb_1) 

print(sum(list(map(phi, Xb_uninfected_test))))

print(sum(list(map(phi, Xb_infected_test))))