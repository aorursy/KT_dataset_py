import matplotlib.pyplot as plt # plotting

from skimage.io import imread # read in images

import numpy as np # linear algebra / matrices
a=imread('../input/scroll.tif')

b=imread('../input/wood.tif')

c=imread('../input/asphalt_gray.tif')
%matplotlib inline

# setup the plotting environment

plt.imshow(a, cmap = 'gray') # show a single image
%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.5,5))

ax1.imshow(a, cmap = 'gray')

ax2.imshow(b, cmap = 'gray')

ax3.imshow(c, cmap = 'gray')
# Identify the region

x1 = 0

x2 = 200

y1 = 800

y2 = 1000



# extract a sub image

subA1=a[x1:x2,y1:y2];

snrA1=np.mean(subA1)/np.std(subA1) # compute the snr

print("SNR for A_1 is {}".format(snrA1))

plt.imshow(subA1)
d=np.mean(imread('../input/testpattern.png'),2)

plt.imshow(d, cmap= 'gray')
from numpy.random import randn



def show_noisy_images(scale_100, scale_10, scale_5, scale_2, scale_1):

    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(8.5,5))

    ax1.imshow(d)

    ax1.set_title('Original')



    d_snr100=d+scale_100*randn(*d.shape);

    ax2.imshow(d_snr100)

    ax2.set_title('SNR 100')



    d_snr10=d+scale_10*randn(*d.shape);

    ax3.imshow(d_snr10)

    ax3.set_title('SNR 10')



    scale = 100 

    d_snr5=d+scale_5*randn(*d.shape);

    ax4.imshow(d_snr5)

    ax4.set_title('SNR 5')



    scale = 1000 

    d_snr2=d+scale_2*randn(*d.shape);

    ax5.imshow(d_snr100)

    ax5.set_title('SNR 2')



    scale = 5000 

    d_snr1=d+scale_1*randn(*d.shape);

    ax6.imshow(d_snr1)

    ax6.set_title('SNR 1')

    return {1: d_snr1, 2: d_snr2, 5: d_snr5, 10: d_snr10, 100: d_snr100}

noisy_images = show_noisy_images(scale_100 = 1.0, 

                  scale_10 = 1.0, 

                  scale_5 = 1.0,

                  scale_2 = 1.0, 

                  scale_1 = 1.0)
from scipy.ndimage.filters import uniform_filter

# Size of the filter window

N=[3,5,7];

# Images

for i,filter_size in enumerate(N):

    fig, all_axes = plt.subplots(5, 2, figsize=(6,22))

    for ((snr,img), (ax1,ax2)) in zip(noisy_images.items(),all_axes):

        ax1.imshow(img, cmap='gray')

        ax1.set_title("Raw, SNR:{}".format(snr))

        ax2.imshow(uniform_filter(img,filter_size), cmap='gray')

        ax2.set_title("N:{}, SNR:{}".format(filter_size,snr))
from scipy.ndimage.filters import median_filter



# Size of the filter window

N=[3,5,7];

# Images

for i,filter_size in enumerate(N):

    fig, all_axes = plt.subplots(5, 2, figsize=(6,22))

    for ((snr,img), (ax1,ax2)) in zip(noisy_images.items(),all_axes):

        ax1.imshow(img, cmap='gray')

        ax1.set_title("Raw, SNR:{}".format(snr))

        break # fix the code to filter an image and show it

        # ax2.imshow(uniform_filter(img,filter_size), cmap='gray')

        # ax2.set_title("N:{}, SNR:{}".format(filter_size,snr))
from six.moves import xrange

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



    for ii in xrange(niter):



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
# apply the filter

anisodiff(noisy_images[100])