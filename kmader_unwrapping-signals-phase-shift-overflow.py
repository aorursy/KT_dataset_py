%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.tri as mtri

from IPython.display import Image, display, SVG, clear_output, HTML

plt.rcParams["figure.figsize"] = (6, 6)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'viridis' # viridis looks better
import numpy as np

from scipy import signal
def compare_signal(raw_signal, **kwargs):

    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(raw_signal, 'k-', label='Raw Signal')

    for i, (sig_name, sig_value) in enumerate(kwargs.items()):

        ax1.plot(sig_value, '-.', lw=4, alpha=0.5, label=sig_name)

    ax1.legend()
simple_1d = np.arange(0, 1024, 4)

simple_1d_wrapped = simple_1d % 255

compare_signal(simple_1d, Wrapped=simple_1d_wrapped)
compare_signal(simple_1d, Wrapped=simple_1d_wrapped, Unwrapped=np.unwrap(simple_1d_wrapped, discont=127))
complex_1d = 768*np.sin(np.linspace(0, np.pi, 100))

complex_1d_wrapped = complex_1d % 255

compare_signal(complex_1d, Wrapped=complex_1d_wrapped, Unwrapped=np.unwrap(complex_1d_wrapped, discont=127))
np.random.seed(2019)

noisy_1d = complex_1d+np.random.uniform(-50, 50, size=complex_1d.shape)

noisy_1d_wrapped = noisy_1d % 255

compare_signal(noisy_1d, Wrapped=noisy_1d_wrapped, Unwrapped=np.unwrap(noisy_1d_wrapped, discont=127))
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

def compare_2d(raw_image, **kwargs):

    fig, m_axs = plt.subplots(1, 2+len(kwargs), figsize=(8*(2+len(kwargs)), 8))

    r_img_ax = m_axs[0].imshow(raw_image)

    m_axs[0].set_title('Raw Image')

    m_axs[0].axis('off')

    clim = r_img_ax.get_clim()

    for c_ax, (sig_name, sig_value) in zip(m_axs[1:], kwargs.items()):

        c_ax.imshow(sig_value, vmin=clim[0], vmax=clim[1])

        c_ax.set_title(sig_name.replace('_', ' '))

    m_axs[-1].axis('off')

    plt.colorbar(r_img_ax)
simple_2d = 512*xx*yy

simple_2d_wrapped = simple_2d % 255

compare_2d(simple_2d, Wrapped=simple_2d_wrapped, Unwrapped=np.unwrap(simple_2d_wrapped, discont=127))
complex_2d = 768*(1-4*np.abs(0.5-xx)*np.abs(0.5-yy))

complex_2d_wrapped = complex_2d % 255

compare_2d(complex_2d, Wrapped=complex_2d_wrapped, Unwrapped=np.unwrap(complex_2d_wrapped, discont=127))
Unwrapped_X=np.unwrap(complex_2d_wrapped, discont=127, axis=0)

Unwrapped_Y=np.unwrap(complex_2d_wrapped, discont=127, axis=1)

compare_2d(complex_2d, 

           Unwrapped_X=Unwrapped_X,

          Unwrapped_Y=Unwrapped_Y)
compare_2d(complex_2d, 

           Unwrapped_X=Unwrapped_X,

          Unwrapped_Y=Unwrapped_Y,

          Unwrapped_Avg = 0.5*Unwrapped_X+0.5*Unwrapped_Y)
Offset_X = Unwrapped_X-complex_2d_wrapped

Offset_Y = Unwrapped_Y-complex_2d_wrapped

Offset_Max = np.max(np.stack([Offset_X, Offset_Y],0), axis=0)

compare_2d(complex_2d, 

           Offset_X=Offset_X,

           Offset_Y=Offset_Y,

          Offset_Max=Offset_Max,

          Unwrapped_Max=Offset_Max+complex_2d_wrapped)
import doctest

import copy

import functools

from itertools import cycle

def autotest(func):

    globs = copy.copy(globals())

    globs.update({func.__name__: func})

    doctest.run_docstring_examples(

        func, globs, verbose=True, name=func.__name__)

    return func

@autotest

def unwrap(p, discont=127, axis=0, njumps_up=None, njumps_down=None):

    """A more generic unwrap function

    Parameters

    ----------

    p : array_like

        Input array.

    discont : float, optional

        Maximum discontinuity between values, default is ``127``.

    axis : int, optional

        Axis along which unwrap will operate, default is the last axis.

    njumps : int, optional

        Maximum number of discrepencies to correct



    >>> unwrap([0, 1, 2, 0], discont=1.5)

    array([0., 1., 2., 3.])

    >>> unwrap([2, 1, 0, 2], discont=1.5)

    array([ 2.,  1.,  0., -1.])

    >>> unwrap([0, 1, 2, 0, 1, 2, 0], discont=1.5)

    array([0., 1., 2., 3., 4., 5., 6.])

    >>> unwrap([0, 1, 2, 0, 1, 2, 0], discont=1.5, njumps_up=1)

    array([0., 1., 2., 3., 4., 5., 3.])

    >>> unwrap([2, 1, 0, 2], discont=1.5, njumps_down=0)

    array([2., 1., 0., 2.])

    """

    p = np.asarray(p)

    out = np.array(p, copy=True, dtype='d')

    # find the jumps

    dd = np.diff(p, axis=axis)

    ph_correct = np.zeros_like(dd)

    # undo the points that are too extreme

    

    if (njumps_up is not None) or (njumps_down is not None):

        ph_update = np.zeros(dd.shape)

        for i, idx in enumerate(zip(*np.where(dd<-discont))):

            if (njumps_up is not None) and (i>=njumps_up):

                break

            ph_update[idx] = 2*discont

        

        for i, idx in enumerate(zip(*np.where(dd>discont))):

            if (njumps_down is not None) and (i>=njumps_down):

                break

            ph_update[idx] = -2*discont

            

    else:

        dd[np.abs(dd)<discont] = 0

        ph_update = dd.copy()

        ph_update[dd>0] = -2*discont

        ph_update[dd<0] = 2*discont

    

    # update the right part of the array

    slice1 = [slice(None, None)]*p.ndim     # full slices

    slice1[axis] = slice(1, None)

    out[tuple(slice1)] = p[tuple(slice1)] + ph_update.cumsum(axis)

    return out
Unwrapped_X=unwrap(complex_2d_wrapped, discont=127, axis=0)

Unwrapped_Y=unwrap(complex_2d_wrapped, discont=127, axis=1)

Offset_Max = np.max(np.stack([Unwrapped_X-complex_2d_wrapped, Unwrapped_Y-complex_2d_wrapped],0), axis=0)

compare_2d(complex_2d, 

           Unwrapped_X=Unwrapped_X,

          Unwrapped_Y=Unwrapped_Y,

          Unwrapped_Max=Offset_Max+complex_2d_wrapped)
from tqdm import tqdm

def iterative_max_unwrap(in_img, iterations=1000):

    cur_image = in_img

    for i in tqdm(range(iterations)):

        Unwrapped_X=unwrap(cur_image, discont=127, axis=0, njumps_up=1)

        Unwrapped_Y=unwrap(cur_image, discont=127, axis=1, njumps_up=1)

        Offset_X = Unwrapped_X-cur_image

        Offset_Y = Unwrapped_Y-cur_image

        combo_offset = np.stack([Offset_X, Offset_Y], 0)

        Offset_Max = np.max(combo_offset, axis=0)

        Unwrapped_Max = Offset_Max+cur_image

        if np.allclose(cur_image, Unwrapped_Max, atol=0.25):

            break

        else:

            cur_image = Unwrapped_Max

    return cur_image
compare_2d(complex_2d, 

               Wrapped=complex_2d_wrapped,

               Unwrapped_Iterative_Max=iterative_max_unwrap(complex_2d_wrapped))
complexer_2d = 350*(1+np.sin(3*np.pi*xx+0.1)*np.cos(2*np.pi*yy))

complexer_2d_wrapped = complexer_2d % 255

compare_2d(complexer_2d,

           Wrapped=complexer_2d_wrapped, 

           Unwrapped=np.unwrap(complexer_2d_wrapped, discont=127),

          Unwrapped_Iterative_Max=iterative_max_unwrap(complexer_2d_wrapped, 5000)

          )
def fixwrap(in_img, discont, axis=0):

    slice1 = [slice(None, None)]*in_img.ndim # full slices

    slice1[axis] = slice(0, 1)

    slice1 = tuple(slice1)

    diff_img = np.diff(in_img, n=1, axis=axis, prepend=in_img[slice1])

    diff_img[np.abs(diff_img)>discont] = 0

    diff_img[slice1] = in_img[slice1]

    return diff_img.cumsum(axis)
from numba import jit

@jit(nopython=True)

def conv_cumsum(border_img, dx_img, dy_img, scale_factor=8.0):

    out = np.zeros_like(border_img)

    out[:, 0] = border_img[:, 0]

    out[0, :] = border_img[0, :]

    w_bx = np.abs(dx_img[1:, :]).mean()

    w_by = np.abs(dy_img[:, 1:]).mean()

    for i in range(1, out.shape[0]):

        for j in range(1, out.shape[1]):

            pred_x = out[i-1, j]+dx_img[i,j]

            pred_y = out[i, j-1]+dy_img[i,j]

            pred_xy = out[i-1, j-1]+dx_img[i,j]+dy_img[i,j]

            w_x = np.abs(dx_img[i,j])/w_bx

            w_y = np.abs(dy_img[i,j])/w_by

            out[i,j] = (scale_factor*w_x*pred_x+scale_factor*w_y*pred_y+pred_xy)/(1+scale_factor*w_x+scale_factor*w_y)

    return out
def fixwrap_2d(in_img, discont, scale_factor=8):

    diff_img_list = []

    for axis in range(2):

        slice1 = [slice(None, None)]*in_img.ndim

        slice1[axis] = slice(0, 1)

        slice1 = tuple(slice1)

        diff_img = np.diff(in_img, n=1, axis=axis, prepend=in_img[slice1])

        diff_img[np.abs(diff_img)>discont] = 0

        diff_img[slice1] = 0

        diff_img_list.append(diff_img)

    return conv_cumsum(in_img, diff_img_list[0], diff_img_list[1], scale_factor=scale_factor)
compare_2d(complex_2d,

           Wrapped=complex_2d_wrapped, 

          Fix_Wrap_Function=fixwrap(complex_2d_wrapped, 127, axis=1),

        Fix_Wrap_2D=fixwrap_2d(complex_2d_wrapped, 127, 4)

          )
param_sweep = {f"Fix_Wrap_2D_{a:2.2f}":fixwrap_2d(complex_2d_wrapped, 127, a) for a in np.logspace(-1.5, 3, 8)}

compare_2d(complex_2d,**param_sweep)
compare_2d(complexer_2d,

           Wrapped=complexer_2d_wrapped, 

          Fix_Wrap_Function=fixwrap(complexer_2d_wrapped, 127, axis=1),

        Fix_Wrap_2D=fixwrap_2d(complexer_2d_wrapped, 127, 4)

          )
np.random.seed(2019)

noisy_2d = complex_2d+np.random.uniform(-30, 30, size=complex_2d.shape)

noisy_2d_wrapped = noisy_2d % 255

compare_2d(noisy_2d,

           Wrapped=noisy_2d_wrapped, 

          Fix_Wrap_Function=fixwrap(noisy_2d_wrapped, 127, axis=1),

        Fix_Wrap_2D=fixwrap_2d(noisy_2d_wrapped, 127)

          )
def _xy_cumsum(in_img, discont, verbose=False):

    diff_img_list = []

    for axis in range(2):

        slice1 = [slice(None, None)]*in_img.ndim

        slice1[axis] = slice(0, 1)

        slice1 = tuple(slice1)

        diff_img = np.diff(in_img, n=1, axis=axis, prepend=in_img[slice1])

        diff_img[np.abs(diff_img)>discont] = 0

        diff_img_list.append(diff_img.cumsum(axis))

    if verbose:

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(diff_img_list[0])

        ax2.imshow(diff_img_list[1])

    return diff_img_list



def wrap_avg(in_img, discont, verbose=False):

    int_img_list = _xy_cumsum(in_img, discont, verbose=verbose)

    int_img_list += [c_img[::-1, ::-1] for c_img in _xy_cumsum(in_img[::-1, ::-1], discont, verbose=verbose)]

    int_img_list.append(in_img)

    return np.median(int_img_list, 0)
compare_2d(complexer_2d,

           Wrapped=complexer_2d_wrapped, 

        Fix_Wrap_2D=wrap_avg(complexer_2d_wrapped, 64, True)

          )
from skimage.transform import radon, iradon

theta=np.linspace(0, 180, 360)

rad_data = radon(complexer_2d_wrapped, theta=theta, preserve_range=False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

ax1.imshow(rad_data)

drad_data = np.diff(rad_data,axis=1)

print(drad_data.shape)

drad_data[np.abs(drad_data)>1500] = 0

ax2.imshow(drad_data)

rad_data[:, 1:] = drad_data.cumsum(1)

ax3.imshow(iradon(rad_data, theta=theta))
from skimage.transform import rotate

from scipy.ndimage import zoom

def rot_clean_irot(raw_img, angle, discont):

    rot_args = {'preserve_range': True, 'order': 0, 'mode': 'edge', 'clip': False}

    in_img = rotate(raw_img, angle=angle, **rot_args)

    slice1 = [slice(None, None)]*in_img.ndim

    slice1[0] = slice(0, 1)

    slice1 = tuple(slice1)

    diff_img = np.diff(in_img, n=1, axis=0, prepend=in_img[slice1])

    diff_img[np.abs(diff_img)>discont] = 0

    diff_img[slice1] = in_img[slice1]

    return rotate(diff_img.cumsum(0), angle=-angle, **rot_args)



def multi_angle(in_img, discont, steps=10):

    return np.max([rot_clean_irot(in_img, discont=discont, angle=angle) 

             for angle in np.linspace(0, 270, steps)], 

            axis=0)

compare_2d(complexer_2d,

           Wrapped=complexer_2d_wrapped, 

        Rotate_Clean=rot_clean_irot(complexer_2d_wrapped, angle=15, discont=127),

           Multi_Clean=multi_angle(complexer_2d_wrapped, discont=127, steps=256)

          )
compare_2d(noisy_2d,

           Wrapped=noisy_2d_wrapped, 

          Fix_Wrap_Function=fixwrap(noisy_2d_wrapped, 127, axis=1),

        Fix_Wrap_2D=fixwrap_2d(noisy_2d_wrapped, 127),

           Multi_Clean=multi_angle(noisy_2d_wrapped, discont=127, steps=128)

          )
fig, ((ax1, ax2, ax2i), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(3, 3, figsize=(15, 15))

ax1.imshow(complexer_2d_wrapped, cmap='viridis')

ax1.set_title('Raw Signal')

diff_x = np.diff(complexer_2d_wrapped, axis=1)

ax2.imshow(diff_x, cmap='RdBu')

ax2.set_title('Numerical Derivative')

# remove extremes and reintegrate

diff_x[np.abs(diff_x)>127] = 0



ax2i.imshow(np.cumsum(diff_x, axis=1))

ax2i.set_title('Integrated')



diff_fft_image = np.fft.fft2(complexer_2d_wrapped)

ax3.imshow(np.log(np.abs(np.fft.fftshift(diff_fft_image))), cmap='viridis')

ax3.set_title('FFT2')

_xx, _yy = np.meshgrid(np.linspace(-1, 1, diff_fft_image.shape[0]), 

                     np.linspace(-1, 1, diff_fft_image.shape[1]), 

                     indexing='ij')

rr = np.fft.fftshift(np.sqrt(np.square(_xx)+np.square(_yy)))

cropped_fft_image = (rr<0.9)*diff_fft_image

diff_fft_image = cropped_fft_image*(1j*np.clip(np.fft.fftshift(_xx), 1e-3, 10))

ax4.set_title('Filtered, Differentiated Spectra')

ax4.imshow(np.log(np.abs(np.fft.fftshift(diff_fft_image))), cmap='viridis')

int_fft_image = diff_fft_image/(1j*np.clip(np.fft.fftshift(_xx), 1e-3, 10))

int_image = np.fft.ifft2(int_fft_image)

ax5.imshow(np.abs(int_image), cmap='viridis')

ax5.set_title('Integrated')





ax7.imshow(np.log(np.abs(np.fft.fftshift(int_fft_image))), cmap='viridis')

ax7.set_title('Filtered Spectra')
conv_cumsum.inspect_types()