help()
import numpy as np
import matplotlib.pyplot as plt

N = 10
np.random.seed(1)
x1 = np.random.rand(N).round(2)+0.5
x2 = np.random.rand(N).round(2)+3.5
x = np.vstack((x1,x2))
x = x.reshape(N*2,)
y1 = np.random.rand(N).round(2)+0.5
y2 = np.random.rand(N).round(2)+1.5
y = np.vstack((y1,y2))
y = y.reshape(N*2,)
plt.scatter(x,y)
plt.show()