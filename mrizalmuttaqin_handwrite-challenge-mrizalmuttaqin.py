import cv2
cv2.__version__
import numpy as N
import pandas as P
import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('/kaggle/input/handwritten/handwritting.jpg')
imgry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h,w = imgry.shape
gr = N.zeros([h,w], dtype=N.uint8)
t = 65
for y in range(0, h):
  for x in range(0, w):
    v = int(imgry[y,x])
    if v < t:
      img[y,x] = 255
    elif v >= t:
      img[y, x] = -255


plt.imshow(img,'gray')
plt.show()