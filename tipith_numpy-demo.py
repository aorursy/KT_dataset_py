import matplotlib
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['figure.figsize'] = (13.0, 10.0)
# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.html

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr)
arr2 = arr.copy().reshape((3, 3))
print(arr2)

p1, rw1 = arr.__array_interface__['data']
print('id={}, data_addr={}, rw={}, dtype={}'.format(hex(id(arr)), hex(p1), rw1, arr.dtype))

p2, rw2 = arr.__array_interface__['data']
print('id={}, data_addr={}, rw={}, dtype={}'.format(hex(id(arr2)), hex(p2), rw2, arr2.dtype))

arr2[0,0] = 1000
print(arr2)
print(arr)
!wget https://devecto.com/wp-content/uploads/2018/02/ryhma_slideriin_ropelit.jpg -O img.jpg

img = imread('img.jpg')
img.shape
# indexing can be done via slicing
heads = img[0:250,1000:1700,:]
print(heads.shape)
plt.imshow(heads)
None
# create monochrome by averaging channels. this shows the usage of reduction api.
# other similar functions such as np.std(), np.sum(), np.min(), np.max() work exactly the same way
_img = np.mean(img, axis=2)
print(_img.shape)
plt.imshow(_img, cmap=plt.get_cmap('gray'))
None

# when not indicating the axis of operation, total array is aggregated upon
print(np.mean(img))
# array assignment via indexing
x = 200
y = 200
img_overlay = img.copy()
img_overlay[x:x+heads.shape[0], y:y+heads.shape[1]:, :] += heads
plt.imshow(img_overlay)
None
# element-wise mathematical operations and array broadcasting

# normalize image to range -1 .. 1 with standard deviation of 1
img_norm = (img - np.mean(img)) / np.std(img)
print('old:  range {:.1f} ... {:.1f}, std {:.1f}'.format(np.min(img), np.max(img), np.std(img)))
print('norm: range {:.1f} ... {:.1f}, std {:.1f}'.format(np.min(img_norm), np.max(img_norm), np.std(img_norm)))

vec = np.linspace(start=1.0, stop=0.0, num=img.shape[0])
print('{:>20s} {}'.format('vec:', vec.shape))

# vec is is explicitly expanded to a shape (650, 1, 1)
vec_explicit = vec[:, np.newaxis, np.newaxis]
print('{:>20s} {}'.format('vec_explicit:', vec_explicit.shape))

_img = img_norm * vec_explicit
print('{:>20s} {}'.format('_img:', _img.shape))

# this is equivalent but rotates the image to match the dimensions to allow broadcasting
_img2 = img_norm.T * vec
print('{:>20s} {}'.format('img_norm.T:', img_norm.T.shape))
print('{:>20s} {}'.format('_img2:', _img2.shape))
print('{:>20s} {}'.format('_img2:', _img2.shape))

plt.imshow(_img2.T)
None
# quick bonus from https://matplotlib.org/gallery/showcase/mandelbrot.html#sphx-glr-gallery-showcase-mandelbrot-py

from matplotlib import colors
import time

def mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = np.linspace(xmin, xmax, xn).astype(np.float32)
    Y = np.linspace(ymin, ymax, yn).astype(np.float32)
    C = X + Y[:, None] * 1j
    N = np.zeros_like(C, dtype=int)
    Z = np.zeros_like(C)
    for n in range(maxiter):
        I = np.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter-1] = 0
    return Z, N

def mandelbrot_image():
    xmin, xmax, xn = -2.25, +0.75, 3000/2
    ymin, ymax, yn = -1.25, +1.25, 2500/2
    maxiter = 200
    horizon = 2.0 ** 40
    log_horizon = np.log(np.log(horizon))/np.log(2)
    Z, N = mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon)

    # This line will generate warnings for null values but it is faster to
    # process them afterwards using the nan_to_num
    with np.errstate(invalid='ignore'):
        M = np.nan_to_num(N + 1 - np.log(np.log(abs(Z)))/np.log(2) + log_horizon)

    dpi = 72
    width = 10
    height = 10*yn/xn
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)
    plt.imshow(M, extent=[xmin, xmax, ymin, ymax], cmap='hot', interpolation="bicubic", norm=colors.PowerNorm(0.3))
    ax.set_xticks([])
    ax.set_yticks([])
mandelbrot_image()
