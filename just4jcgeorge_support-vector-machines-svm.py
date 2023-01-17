from IPython.display import Image

import os

Image("../input/week4images/SVM1.jpeg",width="800", height="800")
Image("../input/week4images/SVM2.png",width="800")
Image("../input/week4images/SVM3.png",width="800")
from IPython.display import YouTubeVideo



YouTubeVideo('efR1C6CvhmE', width=800, height=300)
import numpy as np

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import samples_generator



x, y = samples_generator.make_blobs(n_samples=60, centers=2, random_state=30, cluster_std=0.8) # Generate samples



plt.figure(figsize=(10, 8)) # Plot

plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')
plt.figure(figsize=(10, 8))

plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')



# Draw three split lines

x_temp = np.linspace(0, 6)

for m, b in [(1, -8), (0.5, -6.5), (-0.2, -4.25)]:

    y_temp = m * x_temp + b

    plt.plot(x_temp, y_temp, '-k')
plt.figure(figsize=(10, 8))

plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')



# Draw three split lines

x_temp = np.linspace(0, 6)

for m, b, d in [(1, -8, 0.2), (0.5, -6.5, 0.55), (-0.2, -4.25, 0.75)]:

    y_temp = m * x_temp + b

    plt.plot(x_temp, y_temp, '-k')

    plt.fill_between(x_temp, y_temp - d, y_temp + d, color='#f3e17d', alpha=0.5)
Image("../input/week4images/SVM4.png",width="800")
from sklearn.svm import SVC



linear_svc = SVC(kernel='linear')

linear_svc.fit(x, y)
linear_svc.support_vectors_
def svc_plot(model):

    

    # Get the current axes submap data and prepare for drawing the split line

    ax = plt.gca()

    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)

    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)

    Y, X = np.meshgrid(y, x)

    xy = np.vstack([X.ravel(), Y.ravel()]).T

    P = model.decision_function(xy).reshape(X.shape)

    

    # Draw a dividing line using the outline method

    ax.contour(X, Y, P, colors='green', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

    

    # Mark the location of the support vector

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], c='green', s=100)
# Draw maximum separation support vector diagram



plt.figure(figsize=(10, 8))

plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')

svc_plot(linear_svc)
x = np.concatenate((x, np.array([[3, -4], [4, -3.8], [2.5, -6.3], [3.3, -5.8]])))

y = np.concatenate((y, np.array([1, 1, 0, 0])))



plt.figure(figsize=(10, 8))

plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')
linear_svc.fit(x, y) # Train



# Plot

plt.figure(figsize=(10, 8))

plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')

svc_plot(linear_svc)
from ipywidgets import interact

import ipywidgets as widgets



def change_c(c):

    linear_svc.C = c

    linear_svc.fit(x, y)

    plt.figure(figsize=(10, 8))

    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')

    svc_plot(linear_svc)

    

interact(change_c, c=[1, 10000, 1000000])
Image("../input/week4images/SVM5.jpeg",width="800")
x2, y2 = samples_generator.make_circles(150, factor=.5, noise=.1, random_state=30) # Generate samples



plt.figure(figsize=(8, 8)) # Plot

plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=40, cmap='bwr')
def kernel_function(xi, xj):

    poly = np.exp(-(xi**2 + xj**2))

    return poly
from mpl_toolkits import mplot3d

from ipywidgets import interact, fixed



r = kernel_function(x2[:,0], x2[:,1])

plt.figure(figsize=(10, 8))

ax = plt.subplot(projection='3d')

ax.scatter3D(x2[:, 0], x2[:, 1], r, c=y2, s=40, cmap='bwr')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('r')
rbf_svc = SVC(kernel='rbf')

rbf_svc.fit(x2, y2)
plt.figure(figsize=(8, 8))

plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=40, cmap='bwr')



svc_plot(rbf_svc)
def change_c(c):

    rbf_svc.C = c

    rbf_svc.fit(x2, y2)

    plt.figure(figsize=(8, 8))

    plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=40, cmap='bwr')

    svc_plot(rbf_svc)

    

interact(change_c, c=[1, 100, 10000])