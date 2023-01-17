###construct a non linear decision boundary####
import numpy as np 
import sklearn 
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles
X,y = make_circles(90, factor=0.2, noise=0.1) 
plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap='seismic')
plt.show()