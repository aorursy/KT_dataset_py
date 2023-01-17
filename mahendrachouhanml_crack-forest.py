import matplotlib.pyplot as plt

import numpy as np

import cv2
ino = 2

img = cv2.imread(f'../input/crackforest/Images/{ino:03d}.jpg').transpose(2,0,1).reshape(1,3,320,480)

mask = cv2.imread(f'../input/crackforest/Masks/{ino:03d}_label.PNG')
# Plot the input image, ground truth and the predicted output

plt.figure(figsize=(10,10));

plt.subplot(131);

plt.imshow(img[0,...].transpose(1,2,0));

plt.title('Image')

plt.axis('off');

plt.subplot(132);

plt.imshow(mask);

plt.title('Ground Truth')

plt.axis('off');



plt.axis('off');

# plt.savefig('./SegmentationOutput.png',bbox_inches='tight')