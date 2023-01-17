import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import imageio
from skimage.transform import resize

f = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0]])
f = np.pad(f, 6, 'constant', constant_values=0)
f = np.repeat(f, 4, axis=0)
f = np.repeat(f, 4, axis=1)
# globals go here
figsize = (6,6)
# f = imageio.imread("../input/Fig5.26a.jpg").astype(float)
# f = resize(f, (100,100), mode='edge')
N, M = f.shape

plt.imshow(f, cmap='gray')
plt.title("input image")
plt.show()

lapl = np.array([[1, 1, 1],
                 [1,-8, 1],
                 [1, 1, 1]])

g = convolve2d(f, lapl, mode='same')

plt.imshow(g, cmap='gray')
plt.title("laplacian of image")
plt.colorbar()
plt.show()

crossings = np.zeros((N, M))

for i in range(N-3):
    for j in range(M-3):
        # decide if there is a zero crossing
        p11 = g[i, j]
        p12 = g[i, j+1]
        p21 = g[i+1, j]
        p22 = g[i+1, j+1]
        
        # Ignoring horizontal/vertical edges because lazy
        
        # diagonal edges
        if (p11 > 0 and p22 < 0) or (p11 < 0 and p22 > 0):
            crossings[i, j] = 1
            crossings[i+1, j+1] = 1
        if (p11 > 0 and p22 < 0) or (p11 < 0 and p22 > 0):
            crossings[i, j+1] = 1
            crossings[i+1, j] = 1
                
plt.imshow(crossings, cmap='gray')
plt.title("zero crossings (not all, just diagonal ones)")
plt.colorbar()
plt.show()
