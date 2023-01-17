import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import data
from sklearn.cluster import KMeans
# Loading image
astronaut = data.astronaut()
# Normalisation (System RGB)
astronaut_norm = np.array(astronaut, dtype=np.float64) / 255
# Show image
# without normalisation, imshow method doesn't work
plt.imshow(astronaut_norm)
# Visualization of pixel values
# Reshape from 512x512x3 to 262144x3
data = astronaut_norm.reshape(-1,3)
# Conversation from array to pandas Dataframe
dataframe = pd.DataFrame(data=data[0:, 0:], columns=['R', 'G', 'B'])
# First five pixels
dataframe.head()
# Nombre of colors in image
dataframe.drop_duplicates().shape
n_colors = 64
# Model creation
# random_state to obtain the same results on each compilation
kmeans = KMeans(n_clusters = n_colors, random_state = 0)
kmeans.fit(data)
# Image compression
img64 = kmeans.cluster_centers_[kmeans.labels_]
# To show image reshape it to 64x64x3
img64 = img64.reshape(astronaut_norm.shape)
plt.imshow(img64)
# visualization of pixels values in RGB system
data64 = img64.reshape(-1,3)
dataframe64 = pd.DataFrame(data=data64[0:, 0:], columns=['R', 'G', 'B'])
dataframe64.head() # First five pixels
# Number of colors in the compressed image
dataframe64.drop_duplicates().shape
# Now, image of 16 colors!
kmeans_ = KMeans(n_clusters = 16, random_state = 0).fit(data)
img16 = kmeans_.cluster_centers_[kmeans_.labels_]
img16 = img16.reshape(astronaut_norm.shape)
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(astronaut_norm)
ax1.set_title('Original image')
ax1.axis('off')
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(img64)
ax2.set_title('Image of 64 colors')
ax2.axis('off')
ax3 = fig.add_subplot(1,3,3)
ax3.imshow(img16)
ax3.set_title('Image of 16 colors')
ax3.axis('off')