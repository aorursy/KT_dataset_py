import numpy as np
import matplotlib.pyplot as plt
from numpy import copy
from skimage.segmentation import find_boundaries
from PIL import Image
# Read in an example mask and view the critical part of it
msk = np.asarray(Image.open("../input/train_label/171206_033642600_Camera_5_instanceIds.png"))
plt.figure(figsize=(20,20))
plt.imshow(msk[1600:1900:, 1000:2100:])
## use find_boundaries function; eyeball check to see how it did (which is remarkably well)
boundaries = find_boundaries(copy(msk), mode = 'thick')
plt.figure(figsize=(20,20))
plt.imshow(boundaries[1600:1900:, 1000:2100:])
# since the `boundaries` array is a boolean, it is simple to use it to set pixels to 255 on the original mask where the bool is True
msk_boundaries = copy(msk)
msk_boundaries[boundaries] = 255

plt.figure(figsize=(20,20))
plt.imshow(msk_boundaries[1600:1900:, 1000:2100:])