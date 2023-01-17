import random

import numpy as np

from matplotlib import  pyplot as plt



xyc = list()

for _ in range(1000):

    # THis is the random sampling of points. Ie random input

    x = (random.random()-0.0)*10

    y = (random.random()-0.0)*10



    # This is show classes are generated

    # ie we use a a parabola. One slide is class 1 and another +1. So seperating surface is 2D parabola



    # We just add noise when we decide class. You can remove noise by reducing this here

    noise =  random.random() * 500



    if (x*0.1)**2 + (y*0.1)**2 + x*y*100 - 1000 + noise > 0:

        c = 1

    else :

        c = -1

    #print(x, y, c, x**2, y**2 , x*y)

    xyc.append([x, y, c])





xyc = np.asarray(xyc)

plt.scatter(xyc[:,0], xyc[:,1], c=xyc[:,2], marker=".")

plt.show()






