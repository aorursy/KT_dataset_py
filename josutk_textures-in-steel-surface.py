import numpy as np # linear algebra

import pandas as pd

pd.set_option("display.max_rows", 101)

import os

import cv2

import json

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams["font.size"] = 15

import seaborn as sns

from collections import Counter

from PIL import Image

import math

import seaborn as sns

from collections import defaultdict

from pathlib import Path

import cv2

from tqdm import tqdm

import warnings

warnings.simplefilter('ignore')
train_df = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")

train_df.head()
import numpy as np

import warnings



def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):

    """

    Anisotropic diffusion.

    Usage:

    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:

            img    - input image

            niter  - number of iterations

            kappa  - conduction coefficient 20-100 ?

            gamma  - max value of .25 for stability

            step   - tuple, the distance between adjacent pixels in (y,x)

            option - 1 Perona Malik diffusion equation No 1

                     2 Perona Malik diffusion equation No 2

            ploton - if True, the image will be plotted on every iteration

    Returns:

            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low

    small intensity gradients are able to block conduction and hence diffusion

    across step edges.  A large value reduces the influence of intensity

    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of

    0.25)

    step is used to scale the gradients in case the spacing between adjacent

    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.

    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:

    P. Perona and J. Malik.

    Scale-space and edge detection using ansotropic diffusion.

    IEEE Transactions on Pattern Analysis and Machine Intelligence,

    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi

    School of Computer Science & Software Engineering

    The University of Western Australia

    pk @ csse uwa edu au

    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal

    Department of Pharmacology

    University of Oxford

    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.

    March 2002 corrected diffusion eqn No 2.

    July 2012 translated to Python

    April 2019 - Corrected for Python 3.7 - AvW 

    """



    # ...you could always diffuse each color channel independently if you

    # really want

    if img.ndim == 3:

        warnings.warn("Only grayscale images allowed, converting to 2D matrix")

        img = img.mean(2)



    # initialize output array

    img = img.astype('float32')

    imgout = img.copy()



    # initialize some internal variables

    deltaS = np.zeros_like(imgout)

    deltaE = deltaS.copy()

    NS = deltaS.copy()

    EW = deltaS.copy()

    gS = np.ones_like(imgout)

    gE = gS.copy()



    # create the plot figure, if requested

    if ploton:

        import pylab as pl

        from time import sleep



        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")

        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)



        ax1.imshow(img,interpolation='nearest')

        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)

        ax1.set_title("Original image")

        ax2.set_title("Iteration 0")



        fig.canvas.draw()



    for ii in range(niter):



        # calculate the diffs

        deltaS[:-1,: ] = np.diff(imgout,axis=0)

        deltaE[: ,:-1] = np.diff(imgout,axis=1)



        # conduction gradients (only need to compute one per dim!)

        if option == 1:

            gS = np.exp(-(deltaS/kappa)**2.)/step[0]

            gE = np.exp(-(deltaE/kappa)**2.)/step[1]

        elif option == 2:

            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]

            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]



        # update matrices

        E = gE*deltaE

        S = gS*deltaS



        # subtract a copy that has been shifted 'North/West' by one

        # pixel. don't as questions. just do it. trust me.

        NS[:] = S

        EW[:] = E

        NS[1:,:] -= S[:-1,:]

        EW[:,1:] -= E[:,:-1]



        # update the image

        imgout += gamma*(NS+EW)



        if ploton:

            iterstring = "Iteration %i" %(ii+1)

            ih.set_data(imgout)

            ax2.set_title(iterstring)

            fig.canvas.draw()

            # sleep(0.01)



    return imgout
palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
def name_and_mask(start_idx):

    col = start_idx

    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]

    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):

        raise ValueError



    labels = train_df.iloc[col:col+4, 1]

    mask = np.zeros((256, 1600, 4), dtype=np.uint8)



    for idx, label in enumerate(labels.values):

        if label is not np.nan:

            mask_label = np.zeros(1600*256, dtype=np.uint8)

            label = label.split(" ")

            positions = map(int, label[0::2])

            length = map(int, label[1::2])

            for pos, le in zip(positions, length):

                mask_label[pos-1:pos+le-1] = 1

            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')

    return img_names[0], mask
train_path = Path("../input/severstal-steel-defect-detection/train_images/")

def show_mask_image(col):

    name, mask = name_and_mask(col)

    img = cv2.imread(str(train_path / name))

    fig, ax = plt.subplots(figsize=(15, 15))



    for ch in range(4):

        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i in range(0, len(contours)):

            cv2.polylines(img, contours[i], True, palet[ch], 2)

    ax.set_title(name)

    ax.imshow(img)

    plt.show()
fig, ax = plt.subplots(1, 4, figsize=(15, 5))

for i in range(4):

    ax[i].axis('off')

    ax[i].imshow(np.ones((50, 50, 3), dtype=np.uint8) * palet[i])

    ax[i].set_title("class color: {}".format(i+1))

fig.suptitle("each class colors")



plt.show()
from IPython.display import display

def anisotropic_diff(name):

    img = cv2.imread(str(train_path / name))

    filter_ = anisodiff(img, niter=5, kappa=10, gamma=0.25, option=1)

    image = scipy.misc.toimage(filter_)

    display(image)

    return image
import numpy as np

import matplotlib.pyplot as plt



import pywt

import pywt.data

import scipy



# Load image

def harr_decomposition(image):

    coeffs2 = pywt.dwt2(image, 'haar')

    LL, (LH, HL, HH) = coeffs2

    print('Haar decomposition Approximation')

    img = scipy.misc.toimage(LL)

    display(img)

    print('Haar decompositon Horizontal detail')

    img = scipy.misc.toimage(LH)

    display(img)

    print('Haar decomposition Vertical detail')

    img = scipy.misc.toimage(HL)

    display(img)

    print('Haar decomposition Diagonal detail')

    img = scipy.misc.toimage(HH)

    display(img)

    return LH
idx_no_defect = []

idx_class_1 = []

idx_class_2 = []

idx_class_3 = []

idx_class_4 = []

idx_class_multi = []

idx_class_triple = []



for col in range(0, len(train_df), 4):

    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]

    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):

        raise ValueError

        

    labels = train_df.iloc[col:col+4, 1]

    if labels.isna().all():

        idx_no_defect.append(col)

    elif (labels.isna() == [False, True, True, True]).all():

        idx_class_1.append(col)

    elif (labels.isna() == [True, False, True, True]).all():

        idx_class_2.append(col)

    elif (labels.isna() == [True, True, False, True]).all():

        idx_class_3.append(col)

    elif (labels.isna() == [True, True, True, False]).all():

        idx_class_4.append(col)

    elif labels.isna().sum() == 1:

        idx_class_triple.append(col)

    else:

        idx_class_multi.append(col)
for idx in idx_no_defect[:5]:

    show_mask_image(idx)

    name, mask = name_and_mask(idx)

    print('Anisotropic Diffusion')

    image=anisotropic_diff(name)

    harr = harr_decomposition(image)
import scipy

for idx in idx_class_1[:5]:

    show_mask_image(idx)

    name, mask = name_and_mask(idx)

    print('Anisotropic Diffusion')

    image=anisotropic_diff(name)

    harr_decomposition(image)
for idx in idx_class_2[:5]:

    show_mask_image(idx)

    name, mask = name_and_mask(idx)

    print('Anisotropic Diffusion')

    image=anisotropic_diff(name)

    harr_decomposition(image)
for idx in idx_class_3[:5]:

    show_mask_image(idx)

    name, mask = name_and_mask(idx)

    image=anisotropic_diff(name)

    harr_decomposition(image)
import scipy.misc

for idx in idx_class_4[:5]:

    show_mask_image(idx)

    name, mask = name_and_mask(idx)

    print('Anisotropic Diffusion')

    image=anisotropic_diff(name)

    harr_decomposition(image)
for idx in idx_class_multi[:5]:

    show_mask_image(idx)

    name, mask = name_and_mask(idx)

    print('Anisotropic Diffusion')

    image=anisotropic_diff(name)

    harr_decomposition(image)
for idx in idx_class_triple:

    show_mask_image(idx)

    name, mask = name_and_mask(idx)

    print('Anisotropic Diffusion')

    image=anisotropic_diff(name)

    harr_decomposition(image)