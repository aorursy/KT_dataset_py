# Libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
images_gray = np.load('../input/l/gray_scale.npy')
images_lab = np.load('../input/ab/ab/ab1.npy')
# gray_scale.npy contains 25k gray scale images and ab1.npy and ab2.npy contains 10k and ab3.npy contains 5k each of ab components
# of LAB color space images.
# So instead of running the code on every 25k images let's just test it on any
# 1 of the images

image_gray = images_gray[0]
image_lab = images_lab[0]
# Recreating RGB image using the L and AB from the dataset

# Initializing with zeros ( or any random number)
img = np.zeros((224, 224, 3))
img[:, :, 0] = image_gray
img[:, :, 1:] = image_lab

# Changing the data type of the img array to 'uint8'
# Refer the above markdown for it.
img = img.astype('uint8')
# Now if you check img values you'd see that the first dimension 
# doesn't have values between 0-1. infact it ranges till 255
img_ = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)


print('Gray scale image')
plt.imshow(image_gray)
plt.show()

print('Recreated image')
plt.imshow(img_)
plt.show()
