# Import libs
import time
import numpy as np
from skimage import io
import matplotlib.animation as animation
import matplotlib.pyplot as plt
%matplotlib inline
# Read the data and transpose the matrix
vol = io.imread("../input/attention-mri.tif")
volume = vol.T
# number of frames
l = np.linspace(0, volume.shape[0], 69)
#Ploting the animation
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):    
    ax1.clear()
    ax1.imshow(volume[i])
    ax1.axis('off')
ani = animation.FuncAnimation(fig,animate,interval=50, frames=68, repeat=False)
# save to gif
ani.save('ani.gif', writer='imagemagick')