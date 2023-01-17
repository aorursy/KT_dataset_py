import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import imageio
from skimage.transform import resize
from scipy.signal import convolve2d

# globals go here
figsize = (6,6)
f_clean = imageio.imread("../input/Fig5.07(a).jpg").astype(float)

"""
This is probably what you expected the noisy image to be
"""
f_dirty_1 = f_clean + np.random.normal(0, 20, size=f_clean.shape)
# np.clip(f_dirty_1, np.min(f_clean), np.max(f_clean), f_dirty_1)

"""
Whereas this is the actual noisy image you were using
"""
f_dirty_2 = imageio.imread("../input/Fig5.07(b).jpg").astype(float)

# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.title("clean image")
# plt.imshow(f_clean, cmap="gray")
# plt.subplot(122)
# plt.title("dirty image (clean + gaussian)")
# plt.imshow(f_dirty_1, cmap="gray")
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.title("clean image")
# plt.imshow(f_clean, cmap="gray")
# plt.subplot(122)
# plt.title("dirty image (read from file)")
# plt.imshow(f_dirty_2, cmap="gray")
# plt.show()

fig = plt.figure(figsize=(8, 8))
plt.title("dirty vs clean histograms")
fig.axes[0].set_yscale("log", nonposy="clip")
plt.hist(f_clean.flatten(), 255, alpha=0.5, label="clean")
plt.hist(f_dirty_1.flatten(), 255, alpha=0.5, label="dirty (clean + gaussian)")
plt.hist(f_dirty_2.flatten(), 255, alpha=0.5, label="dirty (read from file)")
plt.legend()

"""
Notice their histograms have the same shape but are shifted and scaled of each other
"""

plt.show()