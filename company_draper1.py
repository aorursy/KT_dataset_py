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
def bnormalize(mat):
    """much faster brightness normalization, since it's all vectorized"""
    bnorm = np.zeros_like(mat, dtype=np.float32)
    maxes = np.max(mat, axis=1)
    bnorm = mat / np.vstack((maxes, maxes, maxes)).T
    return bnorm
bnorm = bnormalize(msi79_1.matrix)
bnorm_img = msi79_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()
msi79_2 = MSImage(img79_2)
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
plt.figure(figsize=(8,10))
plt.imshow(img79_1)
plt.show()
np.max(img79_1[:,:,0])
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_1[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_2[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()
print(np.min(bnorm79_2_img[:,:,0]))
print(np.max(bnorm79_2_img[:,:,0]))
print(np.mean(bnorm79_2_img[:,:,0]))
print(np.std(bnorm79_2_img[:,:,0]))
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm79_2_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow((bnorm79_2_img[:,:,0] > 0.9999) & \
           (bnorm79_2_img[:,:,1] < 0.9999) & \
           (bnorm79_2_img[:,:,2] < 0.9999))
plt.subplot(122)
plt.imshow(img79_2)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.995)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(bnorm_img[2000, 1000, :])
plt.subplot(122)
plt.plot(img79_1[2000, 1000, :])
from scipy import spatial

pixel = msi79_1.matrix[2000 * 1000, :]
np.shape(pixel)
set144 = [MSImage(plt.imread(smj)) for smj in smjpegs if "set144" in smj]
plt.imshow(set144[0].image)
import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel
# a sobel filter is a basic way to get an edge magnitude/gradient image
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel(set144[0].image[:750,:750,2]))
from skimage.filters import sobel_h

# can also apply sobel only across one direction.
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel_h(set144[0].image[:750,:750,2]), cmap='BuGn')
from sklearn.decomposition import PCA

pca = PCA(3)
pca.fit(set144[0].matrix)
set144_0_pca = pca.transform(set144[0].matrix)
set144_0_pca_img = set144[0].to_matched_img(set144_0_pca)
fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,0], cmap='BuGn')
fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,1], cmap='BuGn')
fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,2], cmap='BuGn')
sub = set144[0].image[:150,:150,2]
def glcm_image(img, measure="dissimilarity"):
    """TODO: allow different window sizes by parameterizing 3, 4. Also should
    parameterize direction vector [1] [0]"""
    texture = np.zeros_like(sub)

    # quadratic looping in python w/o vectorized routine, yuck!
    for i in range(img.shape[0] ):  
        for j in range(sub.shape[1] ):  
          
            # don't calculate at edges
            if (i < 3) or \
               (i > (img.shape[0])) or \
               (j < 3) or \
               (j > (img.shape[0] - 4)):          
                continue  
        
            # calculate glcm matrix for 7 x 7 window, use dissimilarity (can swap in
            # contrast, etc.)
            glcm_window = img[i-3: i+4, j-3 : j+4]  
            glcm = greycomatrix(glcm_window, [1], [0],  symmetric = True, normed = True )   
            texture[i,j] = greycoprops(glcm, measure)  
    return texture
dissimilarity = glcm_image(sub, "dissimilarity")
fig = plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(dissimilarity, cmap="bone")
plt.subplot(1,2,2)
plt.imshow(sub, cmap="bone")
from skimage import color

hsv = color.rgb2hsv(set144[0].image)
fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image, cmap="bone")
plt.subplot(2,2,2)
plt.imshow(hsv[:,:,0], cmap="bone")
plt.subplot(2,2,3)
plt.imshow(hsv[:,:,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:,:,2], cmap='bone')
fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image[:200,:200,:])
plt.subplot(2,2,2)
plt.imshow(hsv[:200,:200,0], cmap="PuBuGn")
plt.subplot(2,2,3)
plt.imshow(hsv[:200,:200,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:200,:200,2], cmap='bone')
fig = plt.figure(figsize=(8, 6))
plt.imshow(hsv[200:500,200:500,0], cmap='bone')
hsvmsi = MSImage(hsv)
import seaborn as sns
sns.distplot(hsvmsi.matrix[:,0], bins=12)
sns.distplot(hsvmsi.matrix[:,1], bins=12)
sns.distplot(hsvmsi.matrix[:,2], bins=12)
plt.imshow(hsvmsi.image[:,:,2] < 0.4, cmap="plasma")
fig = plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(set144[0].image[:250,:250,:])
plt.subplot(1,2,2)
plt.imshow(hsvmsi.image[:250,:250,2] < 0.4, cmap="plasma")
fig = plt.figure(figsize=(8, 8))
img2 = plt.imshow(set144[0].image[:250,:250,:], interpolation='nearest')
img3 = plt.imshow(hsvmsi.image[:250,:250,2] < 0.4, cmap='binary_r', alpha=0.4)
plt.show()