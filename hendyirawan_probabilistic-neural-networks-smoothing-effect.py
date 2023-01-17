import math

def gaussian_pdf1(x, sigma, w):
    return math.exp( -(x - w)**2 / (2 * sigma**2) )
%matplotlib inline
import matplotlib
import numpy as np
# If you get "ImportError: DLL load failed: The specified procedure could not be found."
# see https://github.com/matplotlib/matplotlib/issues/10277#issuecomment-366136451
# Short answer: pip uninstall cntk
import matplotlib.pyplot as plt

sigma = 0.1
w = 0.5
x = np.linspace(w - 2, w + 2, 100)
fig = plt.figure('Fungsi Gaussian')
ax = fig.add_subplot(111)
ax.set_title('Fungsi Gaussian dengan $\sigma = %s, w = %s$' % (sigma, w))
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x; \sigma, w)$')
ax.grid(which='major')
ax.plot(x, [gaussian_pdf1(_, sigma, w) for _ in x])
plt.show()
%matplotlib inline
import matplotlib
import numpy as np
# If you get "ImportError: DLL load failed: The specified procedure could not be found."
# see https://github.com/matplotlib/matplotlib/issues/10277#issuecomment-366136451
# Short answer: pip uninstall cntk
import matplotlib.pyplot as plt

w = 0.5
x = np.linspace(w - 2, w + 2, 100)
fig = plt.figure('Fungsi Gaussian')
ax = fig.add_subplot(111)
ax.set_title('Fungsi Gaussian dengan $\sigma = \{0.1, 0.2, 0.5, 1.0\}; w = %s$' % (w))
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x; \sigma, w)$')
ax.grid(which='major')
ax.plot(x, [gaussian_pdf1(_, 0.1, w) for _ in x], label='$\sigma = 0.1$')
ax.plot(x, [gaussian_pdf1(_, 0.2, w) for _ in x], label='$\sigma = 0.2$')
ax.plot(x, [gaussian_pdf1(_, 0.5, w) for _ in x], label='$\sigma = 0.5$')
ax.plot(x, [gaussian_pdf1(_, 1.0, w) for _ in x], label='$\sigma = 1.0$')
plt.legend()
plt.show()
import pandas as pd

train = pd.read_csv('../input/simpleocr.csv')
train
import math

def gaussian_pdf2(x, sigma, w_j):
    return math.exp(
        -( (x[0] - w_j[0])**2 + (x[1] - w_j[1])**2 ) /
        (2 * sigma**2) )
# Tahap pertama:
# For setiap pola w_j
#   Bentuk unit pola dengan memasukkan vektor bobot w_j
W = train[['length', 'area']].values
print('W = %s' % W)

#   Hubungkan unit pola pada unit penjumlah untuk kelas C_k yang sesuai
def sample_indexes_for_category(C_k):
    matching_samples_index = list(train[train.label == C_k].index)
    print('Samples untuk C_k=%s: %s' % (C_k, matching_samples_index))
    return matching_samples_index

C = train.label.unique()
print('Classes: %s' % C)

sample_indexes_for_category(C[0])
sample_indexes_for_category(C[1])
sample_indexes_for_category(C[2])
# Tentukan konstanta |C_k| untuk setiap unit penjumlah
def count_label(C_k): return len(train[train.label == C_k])

C_count = [count_label(C_k) for C_k in C]
print('C_count: %s' % C_count)
# Tahap kedua:
# For setiap pola w_j
def find_d_j(j):
    # k = indeks kelas w_j
    C_k = train.label[j]
    k = np.where(C==C_k)[0][0]
    # Cari d_j : jarak dengan pola terdekat lain pada kelas k
    sample_indexes = sample_indexes_for_category(C_k)
    sample_indexes.remove(j)
    print('For W[%s]: C_k=%s k=%s. Sample indexes (other): %s' % (j, C_k, k, sample_indexes))
    d_j_list = [np.linalg.norm(W[j] - W[sample_index]) for sample_index in sample_indexes]
    d_j = np.amin(d_j_list) or 1.0
    print('d_j list: %s => d_j = %s' % (d_j_list, d_j))
    return d_j

find_d_j(0)
find_d_j(1)
find_d_j(5)
# d_tot[k] = d_tot[k] + d_j
def find_d_tot(C_k):
    return np.sum(find_d_j(j) for j in sample_indexes_for_category(C_k))

d_tot = np.array([find_d_tot(C_k) for C_k in C])
print('d_tot[0] = %s' % d_tot[0])
print('d_tot[1] = %s' % d_tot[1])
print('d_tot[2] = %s' % d_tot[2])
# Tentukan g (brute force)
g = 2.0

# For setiap kelas k
#   d_avg[k] = d_tot[k] / |C_k|
d_avg = d_tot / C_count
print('d_avg = %s' % d_avg)

#   sigma_k = g . d_avg[k]
sigmas = g * d_avg
print('sigmas = %s' % sigmas)