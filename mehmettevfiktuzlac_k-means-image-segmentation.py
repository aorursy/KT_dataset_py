from matplotlib.image import imread

import matplotlib.pyplot as plt
img = imread("../input/label2/dog.jpg")
plt.figure(figsize=(10,12))

plt.imshow(img)
img = img/255.0 
img.shape
X = img.reshape(-1,3)
X.shape
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=4)

k_means.fit(X)
img_seg = k_means.cluster_centers_
img_seg
k_means.labels_
img_seg = img_seg[k_means.labels_]
img_seg = img_seg.reshape(img.shape)
plt.figure(figsize=(10,12))

plt.imshow(img_seg)