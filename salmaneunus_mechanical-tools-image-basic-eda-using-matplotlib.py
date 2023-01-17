%matplotlib inline
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
img = mpimg.imread('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Hammer/000002.jpg')

print(img)
implot = plt.imshow(img)
lum_img = img[:,:,0]

plt.imshow(lum_img)
plt.imshow(lum_img, cmap="hot")
imgplot = plt.imshow(lum_img)

imgplot.set_cmap('nipy_spectral')
imgplot = plt.imshow(lum_img)

plt.colorbar()

plt.title("Color Bar")
plt.hist(lum_img.ravel(), bins=100, range=(0.0, 1.0), fc='k', ec='k')
img2 = mpimg.imread('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Screw Driver/000007.jpg')

print(img)
img2plot = plt.imshow(img2)
lum_img2 = img2[1:,0:,0]

plt.imshow(lum_img2)
plt.imshow(lum_img2,cmap="hot")
img2plot = plt.imshow(lum_img2)

img2plot.set_cmap('inferno')
img2plot = plt.imshow(lum_img2)

img2plot.set_cmap('plasma')
img2plot = plt.imshow(lum_img2)

plt.colorbar()

plt.title("Color Bar")