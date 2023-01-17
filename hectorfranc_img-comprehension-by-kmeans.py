from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score

from sklearn.datasets import load_sample_image



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
my_img = load_sample_image('china.jpg')
plt.imshow(my_img)

plt.xticks([])

plt.yticks([])

plt.show()
my_img.reshape(-1, 3).shape
model = KMeans(n_clusters=64).fit(my_img.reshape(-1, 3))
predicted_img = model.cluster_centers_[model.labels_].reshape(my_img.shape)
plt.imshow(predicted_img.reshape(my_img.shape)/255.0)