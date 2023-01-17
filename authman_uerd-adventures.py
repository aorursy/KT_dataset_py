!pip install lycon > /dev/null

!git clone https://github.com/dwgoon/jpegio > /dev/null

!pip install ./jpegio/. > /dev/null

!rm -rf jpegio
# The usual suspects

import pandas as pd

import numpy as np

import jpegio as jio

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

import lycon

from scipy import ndimage, fftpack

from numpy import matlib as mb

from glob import glob

from tqdm.contrib.concurrent import process_map
look = lycon.load('../input/alaska2-image-steganalysis/Cover/20015.jpg')

plt.imshow(look)

plt.show()
img = jio.read('../input/alaska2-image-steganalysis/Cover/20015.jpg')

ax = plt.subplots(1,3, figsize=(10,5))[1]

ax[0].imshow(img.coef_arrays[0], vmin=-25, vmax=25, cmap='gray'); ax[0].set_title('Y')

ax[1].imshow(img.coef_arrays[1], vmin=-25, vmax=25, cmap='gray'); ax[0].set_title('Cr')

ax[2].imshow(img.coef_arrays[2], vmin=-25, vmax=25, cmap='gray'); ax[0].set_title('Cb')

plt.show()



plt.title('Histogram of DCT Coefficients')

plt.hist(img.coef_arrays[0].flatten(), 100)

plt.show()
simg = jio.read('../input/alaska2-image-steganalysis/UERD/20015.jpg')

ax = plt.subplots(1,3, figsize=(10,5))[1]

ax[0].imshow(img.coef_arrays[0] - simg.coef_arrays[0], cmap='gray')

ax[1].imshow(img.coef_arrays[1] - simg.coef_arrays[1], cmap='gray')

ax[2].imshow(img.coef_arrays[2] - simg.coef_arrays[2], cmap='gray')

plt.show()



plt.hist((img.coef_arrays[0] - simg.coef_arrays[0]).flatten(), 100)

plt.show()
simg = jio.read('../input/alaska2-image-steganalysis/UERD/20015.jpg')

plt.title(f'{(img.coef_arrays[0] != simg.coef_arrays[0]).sum()} total changed components')

plt.imshow(img.coef_arrays[0] != simg.coef_arrays[0], cmap='gray')

plt.show()
def nz_ac_dct_counts_1d(coef, show=False):

    # Reshape for easy arrangement. In Numpy, reshape and transpose are toll-free.

    # Only the views change, the internal arrangements remain the same:

    simage = coef.reshape(512//8,8,512//8,8).transpose(1,3,0,2).reshape(8,8,-1).copy()

    if show:

        print('Resized Shape:', simage.shape)

    

    # Zero out the DC Components:

    simage[0,0] = 0

    

    # Finally, return the count of the non-zero AC components

    return np.float32((simage != 0).sum())
nz_ac_dct_counts_1d(img.coef_arrays[0], show=True)
nz_ac_dct_counts_1d(simg.coef_arrays[0], show=True)
simg.quant_tables[0]
simg.quant_tables[1]
def uerd(coef, quant, wet_cost):

    eps = 1e-7

    row, col = coef.shape

    blk_row = row // 8

    blk_col = col // 8

    

    # Let's use our reshaping trick so that we can take advantage of

    # Numpy broadcasting. That's much better than MatLab than for-loops.

    dct_energy = np.abs(coef).reshape(blk_row,8, blk_col,8).transpose(1,3,0,2).reshape(8, 8, blk_row*blk_col) / quant.reshape(8,8,1)

    

    # Above we calculated the DCT energy the distance from zero / quant table

    # Below we calculate bulk energy, which is the total energy in an 8x8 DCT square:

    blk_energy = dct_energy.sum(axis=(0,1)).reshape(blk_row,blk_col)



    # Think of this mask as a smoothing filter we convolve against

    # the blk_energy computed above

    msk_energy = np.array([

        [0.25, 0.25, 0.25],

        [0.25, 1.00, 0.25],

        [0.25, 0.25, 0.25],

    ])

    blk_energy = ndimage.correlate(blk_energy, msk_energy, mode='nearest').astype(blk_energy.dtype)

    

    # Above we saw blk_energy comes from dct_energy, which has our quant divided out.

    # Now, we're going to put the quant values back into our numerator.

    # Tile the quant tables at every 8x8 square location over the image:

    numerator_unit = quant

    numerator_unit[0,0] = (numerator_unit[0,1] + numerator_unit[1,2]) / 2

    numerator = mb.repmat(numerator_unit, blk_row, blk_col)

    

    # Our denominator is going to be the value of the total energy of the DCT

    # square, expressed into an 8x8 block rather than the per-dct-coefficient (pixel) values:

    denominator = np.ones((8,8,blk_row,blk_col)) * blk_energy.reshape(1,1,blk_row,blk_col)    

    denominator[denominator == 0.0] = 10*eps

    denominator = denominator.transpose(2,0,3,1).reshape(coef.shape)



    # Cool cool cool cool cool cool....

    rho = numerator / denominator

    rho[rho >= wet_cost] = wet_cost

    rho[np.isnan(rho)] = wet_cost

    rho_p1 = rho.copy()

    rho_m1 = rho.copy()



    # We're now thresholded by wet_cost and have two maps.

    # One cost map for plus1 and one for minus one:

    rho_p1[coef > 1023] = wet_cost

    rho_m1[coef < -1023] = wet_cost



    return rho_p1, rho_m1
WET_COST = 1e10

rho_p1, rho_m1 = uerd(img.coef_arrays[0].copy(), img.quant_tables[0], WET_COST)



(rho_p1 != rho_m1).sum()
ax = plt.subplots(1,2, sharey=True, sharex=True)[1]

ax[0].imshow(look)

ax[1].imshow(rho_p1, cmap='gray')

plt.show()
def ternary_entropyf(pP1, pM1):

    eps = 1e-7 # I just choose this value for fp16 safety..

    

    p0 = 1 - pP1 - pM1

    P = np.array([p0, pP1, pM1]).T

    

    H = -P*np.log2(P)

    H[

        (P<eps) |

        (P > 1-eps)

    ] = 0

    return H.sum()
def calc_lambda(rhoP1, rhoM1, message_length, n):

    l3 = 1e+3

    m3 = message_length + 1

    iterations = 0

    

    while m3 > message_length:

        l3 *= 2

        pP1 = (np.exp(-l3 * rhoP1)) / (1 + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))

        pM1 = (np.exp(-l3 * rhoM1)) / (1 + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))

        

        m3 = ternary_entropyf(pP1, pM1)

        iterations += 1

        if iterations > 10:

            return l3



    l1 = 0

    m1 = n

    lamb = 0



    alpha = message_length / n

    

    # Limit search to 30 iterations and require that relative payload embedded

    # is roughly within 1/1000 of the required relative payload

    while iterations<30 and (m1-m3)/n > alpha/1000:

        lamb = l1 + (l3-l1)*0.5

        pP1 = np.exp(-lamb * rhoP1) / (1 + np.exp(-lamb * rhoP1) + np.exp(-lamb * rhoM1))

        pM1 = np.exp(-lamb * rhoM1) / (1 + np.exp(-lamb * rhoP1) + np.exp(-lamb * rhoM1))

        m2 = ternary_entropyf(pP1, pM1)

        

        if m2 < message_length:

            l3 = lamb

            m3 = m2

                 

        else:

            l1 = lamb

            m1 = m2

                 

        iterations += 1

                 

    return lamb



def pm1_simulator(x, rhoP1, rhoM1, m, seed=42):

    n = x.shape[0] * x.shape[1]

    lamb = calc_lambda(rhoP1, rhoM1, m, n)

    pChangeP1 = np.exp(-lamb * rhoP1) / (1 + np.exp(-lamb * rhoP1) + np.exp(-lamb * rhoM1))

    pChangeM1 = np.exp(-lamb * rhoM1) / (1 + np.exp(-lamb * rhoP1) + np.exp(-lamb * rhoM1))



    np.random.seed(seed)

    randChange = np.random.random(x.shape)

    y = x.copy()

    

    # Take care of the alterations:

    y[randChange < pChangeP1] += 1

    y[

        (randChange >= pChangeP1) &

        (randChange < pChangeP1+pChangeM1)

    ] -= 1

    

    return y
def uerd_run(iname, ALPHA):

    WET_COST = 1e10

    js = jio.read(iname)

    coefc = js.coef_arrays[0]

    quant = js.quant_tables[0]

    

    rho_p1, rho_m1 = uerd(coefc, quant, WET_COST)

    coefs = pm1_simulator(coefc, rho_p1, rho_m1, ALPHA*nz_ac_dct_counts_1d(coefc))

    js.coef_arrays[0] = coefs

    return js
for bpnaACDCT in np.arange(0.1, 0.9, 0.1):

    stego = uerd_run('../input/alaska2-image-steganalysis/UERD/20015.jpg', bpnaACDCT)

    

    # Recall, img is our cover:

    dif = img.coef_arrays[0] != stego.coef_arrays[0]

    plt.figure(figsize=(4,4))

    plt.title(f'Bits Per NZ AC DCT Coeff: {bpnaACDCT}')

    plt.imshow(dif.astype(np.uint8), cmap='gray')

    plt.show()