import numpy as np
img = np.random.randint(0,255,[50,50])
from matplotlib import pyplot as plt

plt.imshow(img, cmap="hot")

plt.show()
from scipy import misc



face = misc.face(gray = True)



def display():

    plt.imshow(face, cmap = plt.cm.gray)

    plt.show()
type(face)
face.shape
display()
face = face[200:800,200:800]
display()
face[face>200] = 255
display()
face[face<100] = 0
display()