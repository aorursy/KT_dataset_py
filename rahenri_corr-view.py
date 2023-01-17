

import numpy as np # linear algebra

import math

import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm



positive = 6879

negative = 1176868

#negative = positive



N = 200



mat = np.zeros(shape=[N, N])



tp = 5700



tn = np.linspace(1100000, negative, N)





fn = positive - tp

fp = negative - tn

den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

num = tp * tn - fp * fn



Z = num / den



fig = plt.figure(figsize=(10, 10))

plt.plot(tn, Z)

plt.show()



#print(mat)

        

        

#plt.imshow(mat)