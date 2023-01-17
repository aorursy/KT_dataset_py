import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/kaggle/input/pothole/pothole.jpeg', cv2.IMREAD_GRAYSCALE)
_, morph = cv2.threshold(img, 85, 255, cv2.THRESH_BINARY_INV)
titles = ['Foto Sebelumnya', 'Foto Morphological Transformations']
images = [img, morph]
for i in range(2) :
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()