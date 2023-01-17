import numpy as np # linear algebra

import os

import numpy as np # linear algebra

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib.animation as animation



images = []

fig = plt.figure(figsize=(8,8))



path = '/kaggle/input/raw-images-for-3d-scanner/laser'

for n in range(0, 800, 25):

    img =  mpimg.imread( os.path.join(path, 'laser%03d.jpg' % n) ) 

    imgplot = plt.imshow(img, animated=True)

    images.append([imgplot])

        

ani = animation.ArtistAnimation(fig, images)

ani.save('data.gif', writer='imagemagick', fps=5)



#ani = animation.ArtistAnimation(fig, images, interval=50, blit=True,

#                                repeat_delay=1000)



plt.show()

from IPython.display import Image

Image("../working/data.gif")