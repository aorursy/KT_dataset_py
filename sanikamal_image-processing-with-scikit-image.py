import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import skimage
import skimage.color

%matplotlib inline
data_dir = '../input'
img_dir = os.path.join(data_dir, 'bee_imgs', 'bee_imgs')
data_csv = os.path.join(data_dir, 'bee_data.csv')
data = pd.read_csv(data_csv)
print('Number of rows:', len(data))
data.head()
data.index
data.describe()
data.info()
data.columns
data["caste"].value_counts()
data["pollen_carrying"].value_counts()
data["subspecies"].value_counts() 
def to_file_path(file_name):
    return os.path.join(img_dir, file_name)

data = data.assign(**{'img': data.file.transform(to_file_path)})
example_img_path = data.iloc[1].img
img = mpimg.imread(example_img_path) # Numpy array
plt.imshow(img)
plt.title('Original image')
img_gray = skimage.color.rgb2gray(img)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale')
from skimage import exposure
from functools import partial

# http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
def plot_img_and_hist(image, axes=None, bins=64, title=None):
    """Plot an image along with its histogram and cumulative histogram.

    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1.5, 1]})
        fig.set_size_inches((8, 4))
        # fig.tight_layout()
    ax_img, ax_hist = axes
    # ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    
    if title is not None:
        ax_img.set_title(title)

    # Display histograms per channel
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    
    plot_hist = partial(ax_hist.hist, bins=bins, histtype='bar', linewidth=2, alpha=0.3, density=True)
    plot_hist(red.ravel(), color='red')
    plot_hist(green.ravel(), color='green')
    plot_hist(blue.ravel(), color='blue')

    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_ylim([0, 3])
    ax_hist.set_yticks([])
    
    ax_hist.set_title('Histogram')

    # Display cumulative distribution
    # img_cdf, bins = exposure.cumulative_distribution(image, bins)
    # ax_cdf.plot(bins, img_cdf, 'r')
    # ax_cdf.set_yticks([])

    return ax_img, ax_hist # , ax_cdf
plot_img_and_hist(img, title='Original image')
low = 0.10  # Pixels with intensity smaller than this will be black
high = 0.90  # Pixels with intensity larger than this will be white
img_rescaled = exposure.rescale_intensity(img, in_range=(low, high))
plot_img_and_hist(img_rescaled, title='Rescaled intensity')
img_eq = exposure.equalize_hist(img)
plot_img_and_hist(img_eq, title='Equalized histogram')
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
plot_img_and_hist(img_adapteq, title='Adaptive histogram equalization')
gamma_corrected_2 = exposure.adjust_gamma(img, 2.0)
plot_img_and_hist(gamma_corrected_2, title='Gamma adjustment, $\gamma=2.0$')
gamma_corrected_05 = exposure.adjust_gamma(img, 0.5)
plot_img_and_hist(gamma_corrected_05, title='Gamma adjustment, $\gamma=0.5$')