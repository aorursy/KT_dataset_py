# import the necessary packages

import matplotlib.pyplot as plt



from skimage.feature import hog

from skimage import exposure



%matplotlib inline


# Dalal and Triggs report that using either 2 x 2 or 3 x 3  cells_per_block  obtains reasonable accuracy in most cases.

file  = "../input/hog/hog/dwayne.jpg"

image = plt.imread(file)



fd, hog_image = hog(image, orientations=15, pixels_per_cell=(20, 20),

                    cells_per_block=(2, 2), visualize=True, block_norm="L1", transform_sqrt=False)

# display the image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)



ax1.axis('off')

ax1.imshow(image, cmap=plt.cm.gray)

ax1.set_title('Input image')



# Rescale histogram for better display

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# display the HOG features

ax2.axis('off')

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)

ax2.set_title('Histogram of Oriented Gradients')
file2  = "../input/hog/hog/arrow16.jpeg"

image2 = plt.imread(file2)



fd, hog_image = hog(image2, orientations=8, pixels_per_cell=(16, 16),

                    cells_per_block=(2, 2), visualize=True, block_norm="L1", transform_sqrt=False)

# display the image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)



ax1.axis('off')

ax1.imshow(image2, cmap=plt.cm.gray)

ax1.set_title('Input image')



# Rescale histogram for better display

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# display the HOG features

ax2.axis('off')

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)

ax2.set_title('Histogram of Oriented Gradients')
