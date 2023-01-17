import pandas as pd
train = pd.read_csv('../input/simpleocr.csv')
train
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
import math

def gaussian_pdf2(x, sigma, w_j):
    return math.exp(
        -( (x[0] - w_j[0])**2 + (x[1] - w_j[1])**2 ) /
        (2 * sigma**2) )
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sigma = 0.1
w_j = (0.5, 0.7)

# Plot source code from: https://stackoverflow.com/a/9170879/122441
x_0_range = np.linspace(w_j[0] - 3*sigma, w_j[0] + 3*sigma, 100)
x_1_range = np.linspace(w_j[1] - 3*sigma, w_j[1] + 3*sigma, 100)
X_0, X_1 = np.meshgrid(x_0_range, x_1_range)
fs = np.array( [gaussian_pdf2((x_0, x_1), sigma, w_j)
                for x_0, x_1 in zip(np.ravel(X_0), np.ravel(X_1))] )
FS = fs.reshape(X_0.shape)

fig = plt.figure('Fungsi Gaussian dengan 2 variabel')
ax = fig.add_subplot(111, projection='3d')
ax.set_title('$\sigma = %s, w_j = (%s, %s)$' % (sigma, w_j[0], w_j[1]))
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('$f(x_0, x_1; \sigma, w_j)$')
ax.plot_surface(X_0, X_1, FS)
plt.show()
gaussian_pdf2(x = (0.2, 0.6),
              sigma = 0.1,
              w_j = [0.5, 0.7])
gaussian_pdf2(x = (0.2, 0.6),
              sigma = 0.1,
              w_j = [0.2, 0.5])
import numpy as np

W = np.array([ (train['length'][d], train['area'][d]) 
              for d in range(len(train))])
W
sigma = 0.1
x = (0.2, 0.6)
patterns = np.array([ gaussian_pdf2(x, sigma, w_j) for w_j in W ])
patterns
# Penjumlahan secara manual
c_blue = patterns[0] + patterns[1]
c_red = patterns[2] + patterns[3]
c_green = patterns[4] + patterns[5] + patterns[6]

print('c_blue = %s' % c_blue)
print('c_red = %s' % c_red)
print('c_green = %s' % c_green)
# Penjumlahan secara umum
c_blue = np.sum(
    (patterns[d] if train['label'][d] == 'BLUE' else 0)
    for d in range(len(train)) )
c_red = np.sum(
    (patterns[d] if train['label'][d] == 'RED' else 0)
    for d in range(len(train)) )
c_green = np.sum(
    (patterns[d] if train['label'][d] == 'GREEN' else 0)
    for d in range(len(train)) )
print('c_blue = %s' % c_blue)
print('c_red = %s' % c_red)
print('c_green = %s' % c_green)
categories = [('BLUE', c_blue),
              ('RED', c_red),
              ('GREEN', c_green)]
categories
best_label = None
best_value = None
for cat in categories:
    if not best_label or cat[1] > best_value:
        best_label = cat[0]
        best_value = cat[1]
        
print('Best label: %s. Best value: %s.' % (best_label, best_value))