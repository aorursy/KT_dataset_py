import numpy as np

from random import randrange

import os

from PIL import Image

import matplotlib.pyplot as plt
filenames = os.listdir('../input/butterfly-dataset/leedsbutterfly/images')
img = Image.open('../input/butterfly-dataset/leedsbutterfly/images/' + filenames[randrange(len(filenames))]).convert('L').resize((256,256))
plt.imshow(img, cmap='gray')
U,S,V = np.linalg.svd(np.array(img), full_matrices=False)

S = np.diag(S)
plt.plot(np.cumsum(S)/np.sum(S))

plt.title('Cumulative Sum of Sigma Matrix')
r = 5

reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]
energy = 0

for i in range(r):

    energy = energy + S[i][i]*S[i][i]

energy = energy / np.sum(np.square(S))

print('The first ' + str(r) + ' columns contained ' + str(energy * 100) + '% of the original energy of the image')
plt.imshow(reconstruction,cmap='gray')
r = 10

reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]
energy = 0

for i in range(r):

    energy = energy + S[i][i]*S[i][i]

energy = energy / np.sum(np.square(S))

print('The first ' + str(r) + ' columns contained ' + str(energy * 100) + '% of the original energy of the image')
plt.imshow(reconstruction,cmap='gray')
r = 25

reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]
energy = 0

for i in range(r):

    energy = energy + S[i][i]*S[i][i]

energy = energy / np.sum(np.square(S))

print('The first ' + str(r) + ' columns contained ' + str(energy * 100) + '% of the original energy of the image')
plt.imshow(reconstruction,cmap='gray')
r = 50

reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]
energy = 0

for i in range(r):

    energy = energy + S[i][i]*S[i][i]

energy = energy / np.sum(np.square(S))

print('The first ' + str(r) + ' columns contained ' + str(energy * 100) + '% of the original energy of the image')
plt.imshow(reconstruction,cmap='gray')
img = Image.open('../input/butterfly-dataset/leedsbutterfly/images/' + filenames[randrange(len(filenames))]).convert('L').resize((256,256))

print('Original')

plt.imshow(img, cmap='gray')
U,S,V = np.linalg.svd(np.array(img), full_matrices=False)

S = np.diag(S)

r = 50

reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]

print('Reconstruction')

plt.imshow(reconstruction,cmap='gray')
img = Image.open('../input/butterfly-dataset/leedsbutterfly/images/' + filenames[randrange(len(filenames))]).convert('L').resize((256,256))

print('Original')

plt.imshow(img, cmap='gray')
U,S,V = np.linalg.svd(np.array(img), full_matrices=False)

S = np.diag(S)

r = 50

reconstruction = U[:,:r] @ S[0:r,:r] @ V[:r,:]

print('Reconstruction')

plt.imshow(reconstruction,cmap='gray')