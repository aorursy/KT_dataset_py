%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

plt.rcParams["figure.figsize"] = (15, 10)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'gray' # grayscale looks better

from itertools import cycle

prop_cycle = plt.rcParams['axes.prop_cycle']

colors = prop_cycle.by_key()['color']
from pathlib import Path

import numpy as np

import doctest

import copy

from skimage.io import imread as imread_raw

def imread(x, as_gray=True):

    c_img = imread_raw(x, as_gray=as_gray)

    if c_img.max()<10:

        c_img = (c_img.astype('float32')*255).clip(0, 255).astype('uint8')

    return c_img

from skimage.util import montage as montage2d

from skimage.color import label2rgb



# tests help notebooks stay managable

def autotest(func):

    globs = copy.copy(globals())

    globs.update({func.__name__: func})

    doctest.run_docstring_examples(

        func, globs, verbose=True, name=func.__name__)

    return func

np.random.seed(2019)

xx = np.stack([np.arange(5)]*5, -1)

yy = xx.T

bins_sample_8bit = np.linspace(0, 255, 8)

sample_img = (25*(xx+yy)+np.random.uniform(-10, 10, size=(5, 5))).astype('uint8')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))

sns.heatmap(sample_img, annot=True,fmt='02d', ax=ax1, cmap='viridis')

ax2.hist(sample_img.ravel(), bins_sample_8bit, label='Original', alpha=1)
test_image_path = Path('..') / 'input' / 'test_images'

all_images = sorted(list(test_image_path.glob('*.jpg')))

print('Found', len(all_images), 'images')
bins_8bit = np.linspace(0, 255, 26)

fig, m_axs = plt.subplots(5, 2, figsize=(15, 10))

for (c_ax, d_ax), c_image_path in zip(m_axs, all_images):

    c_img = imread(c_image_path, as_gray=True)

    c_ax.imshow(c_img)

    d_ax.hist(c_img.ravel(), bins_8bit)
from skimage.exposure import rescale_intensity


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

sns.heatmap(sample_img, annot=True,fmt='02d', ax=ax1, cmap='viridis')

ax1.set_title('Original')

ax1.axis('off')

trans_img = rescale_intensity(sample_img)

trans_name = 'Normalized'

sns.heatmap(trans_img, annot=True,fmt='02d', ax=ax2, cmap='viridis')

ax2.set_title(trans_name)

ax2.axis('off')

ax3.hist(sample_img.ravel(), bins_sample_8bit, label='Original', alpha=1)

ax3.hist(trans_img.ravel(), bins_sample_8bit, label=trans_name, alpha=0.5)

ax3.legend()
fig, m_axs = plt.subplots(5, 2, figsize=(15, 10))

for (c_ax, d_ax), c_image_path in zip(m_axs, all_images):

    c_img = imread(c_image_path)

    n_img = rescale_intensity(c_img)

    c_ax.imshow(n_img)

    c_ax.set_title('Old Range {}\nNew Range {}'.format((c_img.min(), c_img.max()), 

                                                       (n_img.min(), n_img.max())))

    d_ax.hist(n_img.ravel(), bins_8bit)
@autotest

class skl_image_wrapper():

    """Simple wrapper around SKLearn functions for images.

    >>> from sklearn.preprocessing import Normalizer

    >>> ski_norm = skl_image_wrapper(Normalizer())

    >>> np.power(ski_norm.fit_transform(np.eye(2)), 2)

    array([[0.5, 0. ],

           [0. , 0.5]])

    >>> ski_norm.fit_transform(np.ones((1,)))

    Traceback (most recent call last):

       ...

    ValueError: Invalid Input Shape (1,)

    """

    def __init__(self, parent):

        self._parent = parent

    def __getattr__(self, name):

        attr = getattr(self._parent, name)

        if callable(attr):

            def _newfunc(X, *args, **kwargs):

                X_shape = np.shape(X)

                batch_size = X_shape[0]

                fix_rgb = False

                if len(X_shape)==2:

                    # just one 2d image

                    Xt = np.reshape(X, (-1, 1))

                    batch_size = Xt.shape[0]

                elif (len(X_shape)==3) and (X_shape[3]<=3):

                    # rgb image (or hsv or greyscale)

                    Xt = np.reshape(X.swapaxes(0, 2).swapaxes(1, 2), (X_shape[3], -1))

                    fix_rgb = True

                    batch_size = X_shape[3]

                elif len(X_shape) in [3, 4]:

                    # multiple multichannel images

                    # or multiple images in 2D

                    Xt = np.reshape(X, (X_shape[0], -1))

                else:

                    raise ValueError('Invalid Input Shape {}'.format(X_shape))

                

                result = attr(Xt, *args, **kwargs)

                

                if fix_rgb:

                    result = result.swapaxes(0, 1)

                

                if np.prod(np.shape(result))==np.prod(X_shape):

                    return np.reshape(result, X_shape)

                else:

                    return np.reshape(result, (batch_size, -1))

            return _newfunc

        else:

            return attr
from sklearn.preprocessing import Normalizer, MinMaxScaler, RobustScaler

ski_robust_norm = skl_image_wrapper(RobustScaler(quantile_range=(25, 75)))

def robust_norm_func(in_img):

    out_img = ski_robust_norm.fit_transform(in_img)

    return (out_img*127+127).clip(0, 255).astype('uint8')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

sns.heatmap(sample_img, annot=True,fmt='02d', ax=ax1, cmap='viridis')

ax1.set_title('Original')

ax1.axis('off')

trans_img = robust_norm_func(sample_img)

trans_name = 'Robust Normalizer'

sns.heatmap(trans_img, annot=True,fmt='02d', ax=ax2, cmap='viridis')

ax2.set_title(trans_name)

ax2.axis('off')

ax3.hist(sample_img.ravel(), bins_sample_8bit, label='Original', alpha=1)

ax3.hist(trans_img.ravel(), bins_sample_8bit, label=trans_name, alpha=0.5)

ax3.legend()
fig, m_axs = plt.subplots(5, 2, figsize=(15, 10))

for (c_ax, d_ax), c_image_path in zip(m_axs, all_images):

    c_img = imread(c_image_path)

    n_img = robust_norm_func(c_img)

    c_ax.imshow(n_img)

    c_ax.set_title('Old Range {}\nNew Range {}'.format((c_img.min(), c_img.max()), 

                                                       (n_img.min(), n_img.max())))

    d_ax.hist(n_img.ravel(), bins_8bit)
from skimage.exposure import equalize_hist

equalize_hist_8bit = lambda in_img: (equalize_hist(in_img)*255).clip(0, 255).astype('uint8')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

sns.heatmap(sample_img, annot=True,fmt='02d', ax=ax1, cmap='viridis')

ax1.set_title('Original')

ax1.axis('off')

trans_img = equalize_hist_8bit(sample_img)

trans_name = 'Equalized'

sns.heatmap(trans_img, annot=True,fmt='02d', ax=ax2, cmap='viridis')

ax2.set_title(trans_name)

ax2.axis('off')

ax3.hist(sample_img.ravel(), bins_sample_8bit, label='Original', alpha=1)

ax3.hist(trans_img.ravel(), bins_sample_8bit, label=trans_name, alpha=0.5)

ax3.legend()
fig, m_axs = plt.subplots(4, 3, figsize=(15, 10))

for (b_ax, c_ax, d_ax), c_image_path in zip(m_axs, all_images):

    c_img = imread(c_image_path, as_gray=True)

    b_ax.imshow(c_img)

    b_ax.axis('off')

    b_ax.set_title('Old Range {}'.format((c_img.min(), c_img.max())))

    n_img = equalize_hist_8bit(c_img)

    c_ax.imshow(n_img)

    c_ax.axis('off')

    c_ax.set_title('New Range {}'.format((n_img.min(), n_img.max())))

    d_ax.hist(n_img.ravel(), bins_8bit)
from scipy.signal import detrend

def detrend_img(in_img): 

    out_img = detrend(detrend(in_img.astype('float32'), axis=0), axis=1)

    return (255.0*(out_img-out_img.min())/(out_img.max()-out_img.min())).clip(0, 255).astype('uint8')

# 1D example

y = np.random.uniform(-1, 1, size=(30))+np.arange(30)*0.1

fig, ax1 = plt.subplots(1,1, figsize=(8, 4))

ax1.plot(y, label='Raw Signal')

ax1.plot(detrend(y), label='Detrended Signal')

ax1.plot(y-detrend(y), label='Trend (subtracted from raw signal)')

ax1.legend()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

sns.heatmap(sample_img, annot=True,fmt='02d', ax=ax1, cmap='viridis')

ax1.set_title('Original')

ax1.axis('off')

trans_img = detrend_img(sample_img)

trans_name = 'Detrended'

sns.heatmap(trans_img, annot=True,fmt='02d', ax=ax2, cmap='viridis')

ax2.set_title(trans_name)

ax2.axis('off')

ax3.hist(sample_img.ravel(), bins_sample_8bit, label='Original', alpha=1)

ax3.hist(trans_img.ravel(), bins_sample_8bit, label=trans_name, alpha=0.5)

ax3.legend()
fig, m_axs = plt.subplots(4, 3, figsize=(15, 10))

for (b_ax, c_ax, d_ax), c_image_path in zip(m_axs, all_images):

    c_img = imread(c_image_path, as_gray=True)

    b_ax.imshow(c_img)

    b_ax.axis('off')

    b_ax.set_title('Old Range {}'.format((c_img.min(), c_img.max())))

    n_img = detrend_img(c_img)

    c_ax.imshow(n_img)

    c_ax.axis('off')

    c_ax.set_title('New Range {}'.format((n_img.min(), n_img.max())))

    d_ax.hist(n_img.ravel(), bins_8bit)
from cv2 import createCLAHE

little_clahe_obj = createCLAHE(clipLimit=4, tileGridSize=(3, 3))

clahe_obj = createCLAHE(clipLimit=4, tileGridSize=(16, 16))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

sns.heatmap(sample_img, annot=True,fmt='02d', ax=ax1, cmap='viridis')

ax1.set_title('Original')

ax1.axis('off')

trans_img = little_clahe_obj.apply(sample_img)

trans_name = 'CLAHE'

sns.heatmap(trans_img, annot=True,fmt='02d', ax=ax2, cmap='viridis')

ax2.set_title(trans_name)

ax2.axis('off')

ax3.hist(sample_img.ravel(), bins_sample_8bit, label='Original', alpha=1)

ax3.hist(trans_img.ravel(), bins_sample_8bit, label=trans_name, alpha=0.5)

ax3.legend()
fig, m_axs = plt.subplots(4, 3, figsize=(15, 10))

for (b_ax, c_ax, d_ax), c_image_path in zip(m_axs, all_images):

    c_img = imread(c_image_path, as_gray=True)

    b_ax.imshow(c_img)

    b_ax.axis('off')

    b_ax.set_title('Old Range {}'.format((c_img.min(), c_img.max())))

    n_img = clahe_obj.apply(c_img)

    c_ax.imshow(n_img)

    c_ax.axis('off')

    c_ax.set_title('New Range {}'.format((n_img.min(), n_img.max())))

    d_ax.hist(n_img.ravel(), bins_8bit)
import math

import numpy



"""

Original Code at https://github.com/nearlyfreeapps/Rolling-Ball-Algorithm/blob/master/rolling_ball.py

Ported to Python from ImageJ's Background Subtractor.

Only works for 8-bit greyscale images currently.

Does not perform shrinking/enlarging for larger radius sizes.

Based on the concept of the rolling ball algorithm described

in Stanley Sternberg's article,

"Biomedical Image Processing", IEEE Computer, January 1983.

Imagine that the 2D grayscale image has a third (height) dimension by the image

value at every point in the image, creating a surface. A ball of given radius

is rolled over the bottom side of this surface; the hull of the volume

reachable by the ball is the background.

http://rsbweb.nih.gov/ij/developer/source/ij/plugin/filter/BackgroundSubtracter.java.html

"""





def smooth(array, window=3.0):

    """

    Applies a 3x3 mean filter to specified array.

    """

    dx, dy = array.shape

    new_array = numpy.copy(array)

    edgex = int(math.floor(window / 2.0))

    edgey = int(math.floor(window / 2.0))



    for i in range(dx):

        for j in range(dy):

            window_array = array[max(i - edgex, 0):min(i + edgex + 1, dx),

                                 max(j - edgey, 0):min(j + edgey + 1, dy)]

            new_array[i, j] = window_array.mean()

    return new_array





def rolling_ball_float_background(float_array, radius, invert, smoothing,

                                  ball):

    """

    Create background for a float image by rolling a ball over the image

    """

    pixels = float_array.flatten()

    shrink = ball.shrink_factor > 1



    if invert:

        for i in range(len(pixels)):

            pixels[i] = -pixels[i]



    if smoothing:

        smoothed_pixels = smooth(numpy.reshape(pixels, float_array.shape))

        pixels = smoothed_pixels.flatten()



    pixels = roll_ball(ball, numpy.reshape(pixels, float_array.shape))



    if invert:

        for i in range(len(pixels)):

            pixels[i] = -pixels[i]

    return numpy.reshape(pixels, float_array.shape)





def roll_ball(ball, array):

    """

    Rolls a filtering object over an image in order to find the

    image's smooth continuous background.  For the purpose of explaining this

    algorithm, imagine that the 2D grayscale image has a third (height)

    dimension defined by the intensity value at every point in the image.  The

    center of the filtering object, a patch from the top of a sphere having

    radius 'radius', is moved along each scan line of the image so that the

    patch is tangent to the image at one or more points with every other point

    on the patch below the corresponding (x,y) point of the image.  Any point

    either on or below the patch during this process is considered part of the

    background.

    """

    height, width = array.shape

    pixels = numpy.float32(array.flatten())

    z_ball = ball.data

    ball_width = ball.width

    radius = ball_width // 2

    cache = numpy.zeros(width * ball_width)



    for y in range(-radius, height + radius):

        next_line_to_write_in_cache = (y + radius) % ball_width

        next_line_to_read = y + radius

        if next_line_to_read < height:

            src = next_line_to_read * width

            dest = next_line_to_write_in_cache * width

            cache[dest:dest + width] = pixels[src:src + width]

            p = next_line_to_read * width

            for x in range(width):

                pixels[p] = -float('inf')

                p += 1

        y_0 = y - radius

        if y_0 < 0:

            y_0 = 0

        y_ball_0 = y_0 - y + radius

        y_end = y + radius

        if y_end >= height:

            y_end = height - 1

        for x in range(-radius, width + radius):

            z = float('inf')

            x_0 = x - radius

            if x_0 < 0:

                x_0 = 0

            x_ball_0 = x_0 - x + radius

            x_end = x + radius

            if x_end >= width:

                x_end = width - 1

            y_ball = y_ball_0

            for yp in range(y_0, y_end + 1):

                cache_pointer = (yp % ball_width) * width + x_0

                bp = x_ball_0 + y_ball * ball_width

                for xp in range(x_0, x_end + 1):

                    z_reduced = cache[cache_pointer] - z_ball[bp]

                    if z > z_reduced:

                        z = z_reduced

                    cache_pointer += 1

                    bp += 1

                y_ball += 1



            y_ball = y_ball_0

            for yp in range(y_0, y_end + 1):

                p = x_0 + yp * width

                bp = x_ball_0 + y_ball * ball_width

                for xp in range(x_0, x_end + 1):

                    z_min = z + z_ball[bp]

                    if pixels[p] < z_min:

                        pixels[p] = z_min

                    p += 1

                    bp += 1

                y_ball += 1



    return numpy.reshape(pixels, array.shape)





class RollingBall(object):

    """

    A rolling ball (or actually a square part thereof).

    """

    def __init__(self, radius):

        if radius <= 10:

            self.shrink_factor = 1

            arc_trim_per = 24

        elif radius <= 30:

            self.shrink_factor = 2

            arc_trim_per = 24

        elif radius <= 100:

            self.shrink_factor = 4

            arc_trim_per = 32

        else:

            self.shrink_factor = 8

            arc_trim_per = 40

        self.build(radius, arc_trim_per)



    def build(self, ball_radius, arc_trim_per):

        small_ball_radius = ball_radius / self.shrink_factor

        if small_ball_radius < 1:

            small_ball_radius = 1

        rsquare = small_ball_radius * small_ball_radius

        xtrim = int(arc_trim_per * small_ball_radius) / 100

        half_width = int(round(small_ball_radius - xtrim))

        self.width = (2 * half_width) + 1

        self.data = [0.0] * (self.width * self.width)



        p = 0

        for y in range(self.width):

            for x in range(self.width):

                xval = x - half_width

                yval = y - half_width

                temp = rsquare - (xval * xval) - (yval * yval)



                if temp > 0:

                    self.data[p] = float(math.sqrt(temp))

                p += 1
@autotest

def rolling_ball_background(array, radius, light_background=True,

                            smoothing=True):

    """

    Calculates and subtracts background from array.

    Arguments:

    array - uint8 numpy array representing image

    radius - radius of the rolling ball creating the background

    light_background - Does image have light background

    smoothing - Whether the image should be smoothed before creating the

                background.

    >>> in_image = 1-np.eye(4)

    >>> rolling_ball_background(in_image, 2)

    array([[0, 0, 0, 0],

           [0, 0, 0, 0],

           [0, 0, 0, 0],

           [0, 0, 0, 0]], dtype=int8)

    >>> rolling_ball_background(np.ones_like(in_image), 2)

    array([[0, 0, 0, 0],

           [0, 0, 0, 0],

           [0, 0, 0, 0],

           [0, 0, 0, 0]], dtype=int8)

    """

    invert = False

    if light_background:

        invert = True



    ball = RollingBall(radius)

    float_array = array

    float_array = rolling_ball_float_background(float_array, radius, invert,

                                                smoothing, ball)

    background_pixels = float_array.flatten()



    if invert:

        offset = 255.5

    else:

        offset = 0.5

    pixels = numpy.int8(array.flatten())



    for p in range(len(pixels)):

        value = (pixels[p] & 0xff) - (background_pixels[p] + 255) + offset

        if value < 0:

            value = 0

        if value > 255:

            value = 255



        pixels[p] = numpy.int8(value)



    return numpy.reshape(pixels, array.shape)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

sns.heatmap(sample_img, annot=True,fmt='02d', ax=ax1, cmap='viridis')

ax1.set_title('Original')

ax1.axis('off')

trans_img = rolling_ball_background(sample_img, 2)

trans_name = 'Rolling Ball'

sns.heatmap(trans_img, annot=True,fmt='02d', ax=ax2, cmap='viridis')

ax2.set_title(trans_name)

ax2.axis('off')

ax3.hist(sample_img.ravel(), bins_sample_8bit, label='Original', alpha=1)

ax3.hist(trans_img.ravel(), bins_sample_8bit, label=trans_name, alpha=0.5)

ax3.legend()
fig, m_axs = plt.subplots(4, 3, figsize=(15, 10))

for (b_ax, c_ax, d_ax), c_image_path in zip(m_axs, all_images):

    c_img = imread(c_image_path, as_gray=True)

    b_ax.imshow(c_img)

    b_ax.axis('off')

    b_ax.set_title('Old Range {}'.format((c_img.min(), c_img.max())))

    n_img = rolling_ball_background(c_img, 5)

    c_ax.imshow(n_img)

    c_ax.axis('off')

    c_ax.set_title('New Range {}'.format((n_img.min(), n_img.max())))

    d_ax.hist(n_img.ravel(), bins_8bit)