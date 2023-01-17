from skimage import io

from skimage.color import rgb2gray

from skimage.filters.rank import maximum, minimum, gradient

from skimage.morphology import disk

import matplotlib.pyplot as plt
image = rgb2gray(io.imread("http://www.brainfacts.org/-/media/Brainfacts2/In-the-Lab/Tools-and-Techniques/Article-Images/MRI_blackandwhite.png"))
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(20,20))

ax1.imshow(image, cmap="jet")

g = gradient(image, disk(1))

ax2.imshow(g, cmap="jet")