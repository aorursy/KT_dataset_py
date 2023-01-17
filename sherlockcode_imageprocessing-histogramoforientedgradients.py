import matplotlib.pyplot as plt
import skimage
from skimage.feature import hog
from skimage import data,color,io,exposure

img1 = skimage.io.imread("../input/image-rocket2/logo.png")
plt.figure(figsize= (10,9))
plt.imshow(img1, cmap ='gray')

fd, hog_image = hog(img1,pixels_per_cell=(16,16), block_norm= 'L2-Hys', visualize = True, multichannel = True)

#this is the line to define plots in matplot lib
fig ,axes = plt.subplots(1,3, figsize= (10,8), sharex= True, sharey=True)
ax=axes.ravel()
ax[0].imshow(img1,cmap='gray')
ax[0].set_title('Original')

#higer contrat image
hog_image_rescaled= exposure.rescale_intensity(hog_image,in_range=(1,10))

ax[1].imshow(hog_image,cmap='gray')
ax[1].set_title('Hog Image')

ax[2].imshow(hog_image_rescaled , cmap= 'gray')
ax[2].set_title('Hog rescaled image')

plt.tight_layout()
plt.show()
 
