import matplotlib.pyplot as plt

import matplotlib.image as mpimg
im1 = mpimg.imread("../input/petfinder-adoption-prediction/test_images/000c21f80-1.jpg")

plt.imshow(im1)
im2 = mpimg.imread("../input/petfinder-images-no-backg/petfinder_images_no_backg/Test/000c21f80-1.jpg")

plt.imshow(im2)