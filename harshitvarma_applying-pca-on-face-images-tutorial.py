import numpy as np
import pandas as pd
%matplotlib inline
from matplotlib import pyplot as plt
import cv2
# INPUT PATHS:
BASE = '../input/att-database-of-faces/'
img = cv2.imread(BASE + 's1/1.pgm', 0) # '0' for reading grayscale images

IMG_SHAPE = img.shape

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('A random grumpy person')
plt.show()
filepaths = [] # Contains the absolute paths of all the image files
for s_i in os.listdir(BASE): # The folders containing the files are labelled as s1, s2, etc
    if s_i != 'README': # There is also a README file present in the data, this must be ignored
        for filename in os.listdir(BASE + s_i):
            filepaths.append(BASE + s_i + '/' + filename)
df = pd.DataFrame({'filepaths':filepaths})
display(df)
images = []
for filepath in df['filepaths']:
    images.append(cv2.imread(filepath, 0).flatten())
images = np.array(images)
from sklearn.decomposition import PCA

pca = PCA(n_components=0.8) # Retain 80% of the variation
pca.fit(images)
z = pca.components_

fig, axes = plt.subplots(4, 11, figsize = (15, 15))
for (ax,i) in zip(axes.flat, range(z.shape[0])):
    ax.imshow(z[i].reshape(IMG_SHAPE), cmap = 'gray')
    ax.axis('off')
fig.tight_layout(pad = 0)
components = pca.transform(images) # A 400*44 matrix
projections = pca.inverse_transform(components) # A 400*10304 matrix
# First 50 reconstructed images:
fig, axes = plt.subplots(5, 10, figsize = (15, 15))
for (ax,i) in zip(axes.flat, range(projections.shape[0])):
    ax.imshow(projections[i].reshape(IMG_SHAPE), cmap = 'gray')
    ax.axis('off')
    if i >= 49:
        break
fig.tight_layout(pad = 0)
# First 50 actual images:
fig, axes = plt.subplots(5, 10, figsize = (15, 15))
for (ax,i) in zip(axes.flat, range(images.shape[0])):
    ax.imshow(images[i].reshape(IMG_SHAPE), cmap = 'gray')
    ax.axis('off')
    if i >= 49:
        break
fig.tight_layout(pad=0)
# Reconstructing the 0th image:

a_0 = components[0] # The components for the 0th image (that is, a_i1, a_i2,..., a_i44)
A_0 = np.dot(z.T, a_0)

# Displaying the images:
A_0 = A_0.reshape(IMG_SHAPE)
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Reconstructed Image')
plt.imshow(A_0, cmap='gray')
plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Actual Image')
plt.imshow(images[0].reshape(IMG_SHAPE), cmap='gray')
plt.tight_layout(pad = 0)
plt.show()
# We first calculate the means along each dimension of the unlabelled input data
featurewise_means = np.mean(images, axis=0) 

# Mean Normalization:
X = images - featurewise_means
# The reduced dimensionality
k = 44 # 80% of the variance is retained

# Computing the covariance matrix:
sigma = np.cov(X) # This is equal to ((X.T*X)/400)

# Computing the singular value decomposition of the covariance matrix:
U, S, V = np.linalg.svd(sigma)

# Choosing the first k columns of U
U_reduced = U[:, 0:k]

# Computing the k basis image vectors 
Z = np.dot(U_reduced.T, X)
# Displaying the basis image vectors:

fig, axes = plt.subplots(4, 11, figsize = (15, 15))
for (ax,i) in zip(axes.flat, range(Z.shape[0])):
    ax.imshow(Z[i].reshape(IMG_SHAPE), cmap = 'gray')
    ax.axis('off')
fig.tight_layout(pad = 0)
# Reconstructing the 0th image: 

# U_reduced is same as the 'components' matrix used previously
A_0 = np.dot(U_reduced[0], Z)

# Displaying the images:
A_0 = A_0.reshape(IMG_SHAPE)
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Reconstructed Image')
plt.imshow(A_0, cmap='gray')
plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Actual Image')
plt.imshow(images[0].reshape(IMG_SHAPE), cmap='gray')
plt.tight_layout(pad = 0)
plt.show()