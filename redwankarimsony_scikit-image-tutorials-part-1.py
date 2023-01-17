import numpy as np

import matplotlib.pyplot as plt

import matplotlib



from skimage import data



matplotlib.rcParams['font.size'] = 12
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

ax = axes.ravel()



# loading default data from scikit-learn library

images = data.stereo_motorcycle()

ax[0].imshow(images[0])

ax[0].axis(False)

ax[1].imshow(images[1])

ax[1].axis(False)



fig.tight_layout()

plt.show()
images = ('brick', 'camera', 'cell', 'checkerboard',  

          'clock',  'colorwheel','grass','gravel', 

          'hubble_deep_field', 'immunohistochemistry',  'moon', 'page', 

          'retina', 'rocket',  'text', 'shepp_logan_phantom' )



rows =4

cols =4

f, ax = plt.subplots(rows, cols, figsize = (15,12))

i=0



for name in images:

    caller = getattr(data, name)

    image = caller()

    if image.ndim == 2:

        ax[int(i/rows),int(i % rows)].imshow(image, cmap = plt.cm.gray)

    else:

        ax[int(i/rows),int(i % rows)].imshow(image)

    ax[int(i/rows),int(i % rows)].set_title(name)

    ax[int(i/rows),int(i % rows)].axis('off')

    i+=1

plt.show()   

from mpl_toolkits.mplot3d import Axes3D

from skimage.morphology import (square, rectangle, diamond, disk, cube,

                                octahedron, ball, octagon, star)



# Generate 2D  structuring elements.

struc_2d = { "square(15)": square(15),

            "rectangle(15, 10)": rectangle(15, 10),

            "diamond(7)": diamond(7),

            "disk(7)": disk(7),

            "octagon(7, 4)": octagon(7, 4),

            "star(5)": star(5)

}



# Generate 3D  structuring elements.

struc_3d = {"cube(11)": cube(11),

            "octahedron(5)": octahedron(5),

            "ball(5)": ball(5)}



# Visualize the elements.

fig = plt.figure(figsize=(20, 20))



idx = 1

for title, struc in struc_2d.items():

    ax = fig.add_subplot(3, 3, idx)

    ax.imshow(struc, cmap="Paired", vmin=0, vmax=12)

    for i in range(struc.shape[0]):

        for j in range(struc.shape[1]):

            ax.text(j, i, struc[i, j], ha="center", va="center", color="w")

    ax.set_axis_off()

    ax.set_title(title)

    idx += 1



for title, struc in struc_3d.items():

    ax = fig.add_subplot(3, 3, idx, projection=Axes3D.name)

    ax.voxels(struc)

    ax.set_title(title)

    idx += 1



fig.tight_layout()

plt.show()
from scipy import ndimage as ndi

import matplotlib.cm as cm

from skimage import data

from skimage import color

from skimage.util import view_as_blocks





# get astronaut from skimage.data in grayscale

l = color.rgb2gray(data.astronaut())



# size of blocks

block_shape = (4, 4)



# see astronaut as a matrix of blocks (of shape block_shape)

view = view_as_blocks(l, block_shape)



# collapse the last two dimensions in one

flatten_view = view.reshape(view.shape[0], view.shape[1], -1)



# resampling the image by taking either the `mean`,

# the `max` or the `median` value of each blocks.

mean_view = np.mean(flatten_view, axis=2)

max_view = np.max(flatten_view, axis=2)

median_view = np.median(flatten_view, axis=2)



# display resampled images

fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)

ax = axes.ravel()



l_resized = ndi.zoom(l, 2, order=3)

ax[0].set_title("Original rescaled with\n spline interpolation (order=3)")

ax[0].imshow(l_resized, extent=(0, 128, 128, 0),cmap=cm.Greys_r)



ax[1].set_title("Block view with\n local mean pooling")

ax[1].imshow(mean_view, cmap=cm.Greys_r)



ax[2].set_title("Block view with\n local max pooling")

ax[2].imshow(max_view, cmap=cm.Greys_r)



ax[3].set_title("Block view with\n local median pooling")

ax[3].imshow(median_view, cmap=cm.Greys_r)



for a in ax:

    a.set_axis_off()



fig.tight_layout()

plt.show()
from skimage.color import rgb2gray



original = data.astronaut()

grayscale = rgb2gray(original)



fig, axes = plt.subplots(1, 2, figsize=(8, 4))

ax = axes.ravel()



ax[0].imshow(original)

ax[0].set_title("Original")

ax[0].axis(False)

ax[1].imshow(grayscale, cmap=plt.cm.gray)

ax[1].set_title("Grayscale")

ax[1].axis(False)



fig.tight_layout()

plt.show()
from skimage.color import rgb2hsv



rgb_img = data.coffee()

hsv_img = rgb2hsv(rgb_img)

hue_img = hsv_img[:, :, 0]

sat_img = hsv_img[:,:,1]

value_img = hsv_img[:, :, 2]



fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(8, 2))



ax0.imshow(rgb_img)

ax0.set_title("RGB image")

ax0.axis('off')

ax1.imshow(hue_img, cmap='hsv')

ax1.set_title("Hue channel")

ax1.axis('off')

ax2.imshow(sat_img)

ax2.set_title('Saturation\nChannel')

ax2.axis('off')

ax3.imshow(value_img)

ax3.set_title("Value channel")

ax3.axis('off')



fig.tight_layout()

hue_threshold = 0.04

binary_img = hue_img > hue_threshold



fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 3))



ax0.hist(hue_img.ravel(), 512)

ax0.set_title("Histogram of the Hue \nchannel with threshold")

ax0.axvline(x=hue_threshold, color='r', linestyle='dashed', linewidth=2)

ax0.set_xbound(0, 0.12)

ax1.imshow(binary_img)

ax1.set_title("Hue-thresholded image")

ax1.axis('off')



fig.tight_layout()
fig, ax0 = plt.subplots(figsize=(4, 3))



value_threshold = 0.10

binary_img = (hue_img > hue_threshold) | (value_img < value_threshold)



ax0.imshow(binary_img)

ax0.set_title("Hue and value thresholded image")

ax0.axis('off')



fig.tight_layout()

plt.show()
from skimage import exposure

from skimage.exposure import match_histograms



reference = data.coffee()

image = data.chelsea()



matched = match_histograms(image, reference, multichannel=True)



fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),

                                    sharex=True, sharey=True)

for aa in (ax1, ax2, ax3):

    aa.set_axis_off()



ax1.imshow(image)

ax1.set_title('Source')

ax2.imshow(reference)

ax2.set_title('Reference')

ax3.imshow(matched)

ax3.set_title('Matched')



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))





for i, img in enumerate((image, reference, matched)):

    for c, c_color in enumerate(('red', 'green', 'blue')):

        img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')

        axes[c, i].plot(bins, img_hist / img_hist.max())

        img_cdf, bins = exposure.cumulative_distribution(img[..., c])

        axes[c, i].plot(bins, img_cdf)

        axes[c, 0].set_ylabel(c_color)



axes[0, 0].set_title('Source')

axes[0, 1].set_title('Reference')

axes[0, 2].set_title('Matched')



plt.tight_layout()

plt.show()
from skimage.color import rgb2hed

from matplotlib.colors import LinearSegmentedColormap



# Create an artificial color close to the original one

cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])

cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',

                                             'saddlebrown'])

cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',

                                               'white'])



ihc_rgb = data.immunohistochemistry()

ihc_hed = rgb2hed(ihc_rgb)



fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)

ax = axes.ravel()



ax[0].imshow(ihc_rgb)

ax[0].set_title("Original image")



ax[1].imshow(ihc_hed[:, :, 0], cmap=cmap_hema)

ax[1].set_title("Hematoxylin")



ax[2].imshow(ihc_hed[:, :, 1], cmap=cmap_eosin)

ax[2].set_title("Eosin")



ax[3].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)

ax[3].set_title("DAB")



for a in ax.ravel():

    a.axis('off')



fig.tight_layout()
from skimage.exposure import rescale_intensity



# Rescale hematoxylin and DAB signals and give them a fluorescence look

h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))

d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))

zdh = np.dstack((np.zeros_like(h), d, h))



fig = plt.figure()

axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])

axis.imshow(zdh)

axis.set_title("Stain separated image (rescaled)")

axis.axis('off')

plt.show()