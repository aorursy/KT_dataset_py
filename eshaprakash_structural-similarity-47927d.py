%matplotlib inline
import cv2
from skimage.measure import compare_ssim
from skimage import data, img_as_float
import matplotlib.pyplot as plt
import numpy as np
compare_ssim(img_as_float(data.camera()), img_as_float(data.camera()))
compare_ssim(img_as_float(data.camera()), img_as_float(data.grass()))
from skimage.io import imread
import os

shampoo1 = cv2.cvtColor(imread("../input/png-shampoo/shampoo1.png"), cv2.COLOR_BGR2GRAY)
shampoo2 = cv2.cvtColor(imread("../input/png-shampoo/shampoo2.png"), cv2.COLOR_BGR2GRAY)
before = cv2.cvtColor(imread("../input/png-shampoo/shampoo1.png"), cv2.IMREAD_COLOR)
after = cv2.cvtColor(imread("../input/png-shampoo/shampoo2.png"), cv2.IMREAD_COLOR)
plt.imshow(shampoo2, cmap = plt.cm.gray)

(H, W) = shampoo1.shape

shampoo2 = cv2.resize(shampoo2, (W, H))

shampoo2.shape
shampoo1.shape
(score, diff) = compare_ssim(shampoo1, shampoo2, full=True)
print("Image similarity", score)
diff = (diff * 255).astype("uint8")
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

plt.imshow(filled_after)
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(before)
axarr[0,1].imshow(after)
axarr[1,0].imshow(mask)
axarr[1,1].imshow(filled_after)


