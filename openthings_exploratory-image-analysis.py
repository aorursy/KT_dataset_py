import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
smjpegs = [f for f in glob.glob("../input/train_sm/*.jpeg")]
print(smjpegs[:9])
set175 = [smj for smj in smjpegs if "set175" in smj]
print(set175)

first = plt.imread('../input/train_sm/set175_1.jpeg')
dims = np.shape(first)
print(dims)
np.min(first), np.max(first)
pixel_matrix = np.reshape(first, (dims[0] * dims[1], dims[2]))
print(np.shape(pixel_matrix))
#plt.scatter(pixel_matrix[:,0], pixel_matrix[:,1])
_ = plt.hist2d(pixel_matrix[:,1], pixel_matrix[:,2], bins=(50,50))
fifth = plt.imread('../input/train_sm/set175_5.jpeg')
dims = np.shape(fifth)
pixel_matrix5 = np.reshape(fifth, (dims[0] * dims[1], dims[2]))
_ = plt.hist2d(pixel_matrix5[:,1], pixel_matrix5[:,2], bins=(50,50))

_ = plt.hist2d(pixel_matrix[:,2], pixel_matrix5[:,2], bins=(50,50))
plt.imshow(first)
plt.imshow(fifth)
plt.imshow(first[:,:,2] - fifth[:,:,1])
second = plt.imread('../input/train_sm/set175_2.jpeg')
plt.imshow(first[:,:,2] - second[:,:,2])
plt.imshow(second)
# simple k means clustering
from sklearn import cluster

kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix)

dims = np.shape(first)
clustered_img = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img)
plt.imshow(first)
ind0, ind1, ind2, ind3 = [np.where(clustered == x)[0] for x in [0, 1, 2, 3]]
# quick look at color value histograms for pixel matrix from first image
import seaborn as sns
sns.distplot(pixel_matrix[:,0], bins=12)
sns.distplot(pixel_matrix[:,1], bins=12)
sns.distplot(pixel_matrix[:,2], bins=12)
# even subsampling is throwing memory error for me, :p
#length = np.shape(pixel_matrix)[0]
#rand_ind = np.random.choice(length, size=50000)
#sns.pairplot(pixel_matrix[rand_ind,:])
set79 = [smj for smj in smjpegs if "set79" in smj]
print(set79)
img79_1, img79_2, img79_3, img79_4, img79_5 = \
  [plt.imread("../input/train_sm/set79_" + str(n) + ".jpeg") for n in range(1, 6)]
img_list = (img79_1, img79_2, img79_3, img79_4, img79_5)

print("Image " + str(n))
plt.figure(figsize=(8,10))
plt.imshow(img_list[0])
plt.show()
class MSImage():
    """Lightweight wrapper for handling image to matrix transforms. No setters,
    main point of class is to remember image dimensions despite transforms."""
    
    def __init__(self, img):
        """Assume color channel interleave that holds true for this set."""
        self.img = img
        self.dims = np.shape(img)
        self.mat = np.reshape(img, (self.dims[0] * self.dims[1], self.dims[2]))

    @property
    def matrix(self):
        return self.mat
        
    @property
    def image(self):
        return self.img
    
    def to_flat_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form when
        derived image would only have one band."""
        return np.reshape(derived, (self.dims[0], self.dims[1]))
    
    def to_matched_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form."""
        return np.reshape(derived, (self.dims[0], self.dims[1], self.dims[2]))
msi79_1 = MSImage(img79_1)
print(np.shape(msi79_1.matrix))
print(np.shape(msi79_1.img))
bnorm = np.zeros_like(msi79_1.matrix, dtype=np.float32)
for x in range(7219900):
    bnorm[x,:] = msi79_1.matrix[x,:] / float(np.max(msi79_1.matrix[x,:]))
bnorm_img = msi79_1.to_matched_img(bnorm)

plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()
msi79_2 = MSImage(img79_2)

def bnormalize(mat):
    bnorm = np.zeros_like(mat, dtype=np.float32)
    for x in range(np.shape(mat)[0]):
        bnorm[x,:] = mat[x,:] / float(np.max(mat[x,:]))
    return bnorm

bnorm79_2 = bnormalize(msi79_2.matrix)
bnorm79_2_img = msi79_2.to_matched_img(bnorm79_2)
plt.figure(figsize=(8,10))
plt.imshow(bnorm79_2_img)
plt.show()
msinorm79_1 = MSImage(bnorm_img)
msinorm79_2 = MSImage(bnorm79_2_img)

_ = plt.hist2d(msinorm79_1.matrix[:,2], msinorm79_2.matrix[:,2], bins=(50,50))
_ = plt.hist2d(msinorm79_1.matrix[:,1], msinorm79_2.matrix[:,1], bins=(50,50))
_ = plt.hist2d(msinorm79_1.matrix[:,0], msinorm79_2.matrix[:,0], bins=(50,50))
import seaborn as sns
sns.distplot(msinorm79_1.matrix[:,0], bins=12)
sns.distplot(msinorm79_1.matrix[:,1], bins=12)
sns.distplot(msinorm79_1.matrix[:,2], bins=12)
