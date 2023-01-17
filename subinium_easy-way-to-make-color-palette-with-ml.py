import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from sklearn.cluster import DBSCAN, KMeans
import requests

from PIL import Image

from io import BytesIO



# Image read

r = requests.get("https://images.unsplash.com/photo-1494783367193-149034c05e8f?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=80")

img = Image.open(BytesIO(r.content))



# Create Figure Object

fig, ax = plt.subplots(1,1)

ax.imshow(img)



# No ticks

ax.set_xticks([])

ax.set_yticks([])



plt.tight_layout()

plt.show()
data = np.asarray(img, dtype="int32")

print(data.shape)
%%time

from mpl_toolkits.mplot3d import Axes3D



# x, y, z axis and Color

x = data[:,:,0]

y = data[:,:,1]

z = data[:,:,2]

C = list(map(tuple, data.reshape(data.shape[0]*data.shape[1], 3)/255.0))





fig = plt.figure(figsize=(20,10))



# plot 1 : simple scatter plot

ax1 = fig.add_subplot(121,projection='3d')

ax1.scatter(x,y,z)

ax1.set_title('Scatter Plot')



# plot 2 : colored scatter plot

ax2 = fig.add_subplot(122,projection='3d')

ax2.scatter(x,y,z,c=C)

ax2.set_title('Colored Scatter Plot')



plt.show()


x = data[:,:,0] 

y = data[:,:,1]

z = data[:,:,2]

C = list(map(tuple, data.reshape(data.shape[0]*data.shape[1], 3)/255.0))



fig = go.Figure(data=[go.Scatter3d(

    x=x.reshape(1,-1)[0],

    y=y.reshape(1,-1)[0],

    z=z.reshape(1,-1)[0],

    mode='markers',

    marker=dict(

        size=3,

        color=C,

        opacity=0.7

    )

)])



fig.show()
%%time

points = data.reshape(data.shape[0]*data.shape[1], 3)/255.0



kmeans = KMeans(n_clusters=6).fit(points)

kmeans.labels_
%%time 



fig = plt.figure(figsize=(10,15))

for i in range(6):

    ax = fig.add_subplot(3,2,i+1,projection='3d')

    ax.set_xticks([])

    ax.set_yticks([])

    ax.set_zticks([])

    C = list(map(tuple,points * (kmeans.labels_==i).reshape(-1,1)))

    ax.scatter(x,y,z,c=C)

plt.show()

    
fig, ax = plt.subplots(1,2, figsize=(12,12))

ax[0].imshow(img)



# No ticks

ax[0].set_xticks([])

ax[0].set_yticks([])



# color palette with plt.Circle

for i in range(6):

    circle = plt.Circle((0.05, (i+1.4)/8), 0.04, color=(points * (kmeans.labels_==i).reshape(-1,1)).sum(axis=0) / sum((kmeans.labels_==i)))

    ax[1].add_artist(circle)



# make xy scale equal & axis off 

plt.gca().set_aspect('equal', adjustable='box')

plt.axis('off')



plt.tight_layout()

plt.show()
from skimage.color import rgb2hsv



data = rgb2hsv(img)





# x, y, z axis and Color

x = data[:,:,0]

y = data[:,:,1]

z = data[:,:,2]



fig = plt.figure(figsize=(20,10))



# plot 1 : simple scatter plot

ax1 = fig.add_subplot(121,projection='3d')

ax1.scatter(x,y,z)

ax1.set_title('Scatter Plot')



# plot 2 : colored scatter plot

ax2 = fig.add_subplot(122,projection='3d')

ax2.scatter(x,y,z,c=C)

ax2.set_title('Colored Scatter Plot')



plt.show()
%%time

points2 = data.reshape(data.shape[0]*data.shape[1], 3)



kmeans2 = KMeans(n_clusters=6).fit(points2)



fig = plt.figure(figsize=(10,15))

for i in range(6):

    ax = fig.add_subplot(3,2,i+1,projection='3d')

    ax.set_xticks([])

    ax.set_yticks([])

    ax.set_zticks([])

    C = list(map(tuple,points * (kmeans2.labels_==i).reshape(-1,1)))

    ax.scatter(x,y,z,c=C)

plt.show()
fig, ax = plt.subplots(1,2, figsize=(12,12))

ax[0].imshow(img)



# No ticks

ax[0].set_xticks([])

ax[0].set_yticks([])



# color palette with plt.Circle

for i in range(6):

    circle = plt.Circle((0.05, (i+1.4)/8), 0.04, color=(points * (kmeans2.labels_==i).reshape(-1,1)).sum(axis=0) / sum((kmeans2.labels_==i)))

    ax[1].add_artist(circle)



# make xy scale equal & axis off 

plt.gca().set_aspect('equal', adjustable='box')

plt.axis('off')



plt.tight_layout()

plt.show()