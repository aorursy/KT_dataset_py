import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline   

plt.rcParams['image.cmap'] = 'gray'

import scipy.ndimage





def image_generator(*filenames):

    for filename in filenames:

        with open(filename, 'rb') as fp:

            for _ in range(100000):

                yield np.array(np.fromstring(fp.read(401), dtype=np.uint8)[1:], dtype=float)

def img_gen():

    return image_generator(*["../input/snake-eyes/snakeeyes_{:02d}.dat".format(nn) for nn in range(10)])
def plot_analysis(ww):

    for n in range(1, 7):

        gga = scipy.ndimage.filters.gaussian_filter(ww, 2 * n * 1.0)

        ggb = scipy.ndimage.filters.gaussian_filter(ww, n * 1.0)



        xx = ggb - gga

        mm = xx == scipy.ndimage.morphology.grey_dilation(xx, size=(3, 3))



        plt.subplot(3, 6, n)

        plt.imshow(gga)

        plt.axis('off')



        plt.subplot(3, 6, n+6)

        plt.imshow(gga-ggb, vmin=-100, vmax=100, cmap=plt.cm.RdBu)

        plt.axis('off')



        plt.subplot(3, 6, n+12)

        plt.imshow(mm)

        plt.axis('off')
imgs = img_gen()

for k in range(10):

    qq = next(imgs)

    ww = qq.reshape(20, 20)

    plt.figure(figsize=(10, 5))

    plot_analysis(ww)
def find_dice(ww):

    gga = scipy.ndimage.filters.gaussian_filter(ww, 4.0)

    ggb = scipy.ndimage.filters.gaussian_filter(ww, 2.0)

    ggb-gga

    xx = ggb - gga

    mm = xx == scipy.ndimage.morphology.grey_dilation(xx, size=(3, 3))

    mm[0, :] = 0

    mm[-1, :] = 0

    mm[:, 0] = 0

    mm[:, -1] = 0

    return np.nonzero(mm)



plt.figure(figsize=(15, 8))

imgs = img_gen()

for k in range(50):

    qq = next(imgs)

    ww = qq.reshape(20, 20)

    plt.subplot(5, 10, k+1)

    plt.imshow(ww)

    plt.axis('off')

    for (y, x) in zip(*find_dice(ww)):

        plt.plot(x, y, 'ro')