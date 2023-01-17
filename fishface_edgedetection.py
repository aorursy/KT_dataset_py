import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage import feature
#read input data
dataset = pd.read_csv("../input/train.csv")
train = dataset.iloc[:,1:].values
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)

# load image
im = train[1][0];


# define scharr filters
scharr = np.zeros((3,3,2))
scharr[:,:,0] = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
scharr[:,:,1] = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
# Compute the Scharr filter
edges = np.zeros((im.shape[0]+2,im.shape[1]+2,3))
for i in range(0,2):
    edges[:,:,i] = ss.convolve2d(im[:,:],scharr[:,:,i])
    
edges[:,:,2] = np.sqrt(np.square(edges[:,:,0]) + np.square(edges[:,:,1]))
# display results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.binary)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=10)

ax2.imshow(edges[:,:,0], cmap=plt.cm.binary)
ax2.axis('off')
ax2.set_title('Scharr filter, X', fontsize=10)

ax3.imshow(edges[:,:,1], cmap=plt.cm.binary)
ax3.axis('off')
ax3.set_title('Scharr filter, Y', fontsize=10)

ax4.imshow(edges[:,:,2], cmap=plt.cm.binary)
ax4.axis('off')
ax4.set_title('Scharr filter, X+Y', fontsize=10)

fig.tight_layout()

plt.show()
plt.close()