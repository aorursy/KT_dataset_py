import numpy as np # linear algebra



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
path = '/kaggle/input/denoising-dirty-documents-in-32x32-chunks/new_dataset/train/'
import cv2

import matplotlib.pyplot as plt 



img6299 = cv2.imread(path+"6299.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img6299, cmap='gray');
plt.imshow(img6299[4:6:,5:7], cmap='gray');

img6299[4:6,5:7].min()
def denoiseImg(img = None, threshold = 175):

    if img is None:

        return

    return np.where(img >threshold, 255, img)
plt.imshow(denoiseImg(img6299), cmap='gray');
img8537 = cv2.imread(path+"8537.png", cv2.IMREAD_GRAYSCALE)

plt.imshow(img8537, cmap='gray');
plt.imshow(denoiseImg(img8537, 175), cmap='gray');
img20060 = cv2.imread(path+"20060.png", cv2.IMREAD_GRAYSCALE)

plt.imshow(img20060, cmap='gray');
plt.imshow(denoiseImg(img20060), cmap='gray');