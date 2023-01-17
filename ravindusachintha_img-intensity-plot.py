%matplotlib inline



# import opencv

import cv2,os



import matplotlib.pyplot as plt

import numpy as np



# print(os.listdir("../input"))
# load image (default BGR format)

raw_image = cv2.imread('../input/sampleImg.png', cv2.IMREAD_UNCHANGED)



# convert BGR to RGB

RGB_img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)



# print original image

plt.figure(figsize=(10,10))

plt.imshow(raw_image)

plt.show()
# print intensity image

total_intensity = raw_image[:,:,0] + raw_image[:,:,1] + raw_image[:,:,2]

plt.figure(figsize=(10,10))

plt.imshow(total_intensity, cmap='gray', vmin=0, vmax=255)

plt.show()
# counting row average

row_avg = total_intensity.mean(axis=0)



x = row_avg.shape[0]

s = []



# 1D vector creation

for i in range(1, x+1):

  s.append(0.22305*i+430.79)



# print the output plot

plt.figure(figsize=(10,5))

plt.plot(s, row_avg)

plt.show()