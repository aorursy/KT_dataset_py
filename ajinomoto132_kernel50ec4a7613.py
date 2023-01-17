import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display 

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
path = '../input/'
radiance = np.loadtxt(path+'Image_ch1_spectralRadiance.dat', dtype=float)
reflectance = np.loadtxt(path+'Image_ch2_spectralReflectance.dat', dtype=float)
param = np.loadtxt(path+'Image_parameter_insituMeasurements.dat', dtype=float)
labels = np.loadtxt(path+'Image_class_insituLabels.dat', dtype=float)
plt.imshow(radiance)
plt.show()
plt.imshow(reflectance)
plt.show()
fig,ax = plt.subplots(1)
# plot the data
ax.scatter(reflectance[:,:3],radiance[:,:3],c='r')
ax.scatter(reflectance[:,3:],radiance[:,3:],c='b')
confusion_matrix(y_true.ravel(), model.labels_)
my_matrix = np.vstack([reflectance.ravel(),radiance.ravel()]).T
model = KMeans(n_clusters=2, random_state = 2)
model.fit_transform(my_matrix)

y_true = np.zeros((6,6))
y_true[:,3:] = 1

tn, fp, fn, tp = confusion_matrix(y_true.ravel(), model.labels_).ravel()
POD = tp; FAR = fn

k = cohen_kappa_score(y_true.ravel(), model.labels_)
POD/18, FAR/18, k
my_model = PCA(n_components=1)
new_matrix= my_model.fit_transform(my_matrix)

print(my_model.explained_variance_)
print(my_model.explained_variance_ratio_)
model = KMeans(n_clusters=2, random_state = 2)
model.fit_transform(new_matrix, y_true)

tn, fp, fn, tp = confusion_matrix(y_true.ravel(), model.labels_).ravel()
POD = tp; FAR = fn

k = cohen_kappa_score(y_true.ravel(), model.labels_)
POD/18, FAR/18, k

confusion_matrix(y_true.ravel(), model.labels_), k
import torch
import torch.nn as nn

m = nn.Conv1d(6, 6, 3, stride=1, padding=1, bias=True,)
m.weight = nn.Parameter((torch.ones(6, 6, 3))/9)
# m.bias = nn.Parameter((torch.ones(6)))

output_reflectance = m(torch.tensor(reflectance.reshape(1,6,6), dtype=torch.float)).detach().numpy().squeeze(0)
output_radiance = m(torch.tensor(radiance.reshape(1,6,6), dtype=torch.float)).detach().numpy().squeeze(0)
plt.imshow(output_radiance)
plt.show()
plt.imshow(output_reflectance)
plt.show()
my_matrix = np.vstack([output_radiance.ravel(),output_reflectance.ravel()]).T
model = KMeans(n_clusters=2, random_state = 2)
model.fit_transform(my_matrix)

print(confusion_matrix(y_true.ravel(), model.labels_))

k = cohen_kappa_score(y_true.ravel(), model.labels_)
k
reg = LinearRegression()
my_matrix = np.vstack([reflectance.ravel(),radiance.ravel()]).T
reg.fit(my_matrix, param.ravel())
res = reg.predict(my_matrix)
bias = res - param.ravel()
np.mean(bias), np.std(bias)
param.shape