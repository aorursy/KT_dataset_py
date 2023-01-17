import sklearn

import PIL

from matplotlib import image

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from skimage import transform, color

from PIL import Image 

import numpy as np

from os import listdir

from os.path import isfile, join

import os
sample_images_positive = []

i=0

for filename in listdir('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/'):

    if i < 10:

        img = plt.imread('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/' + filename)

        sample_images_positive.append(img)

        i+=1
sample_images_negative = []

i=0

for filename in listdir('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/'):

    if i < 10:

        img = plt.imread('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/' + filename)

        sample_images_negative.append(img)

        i+=1
plt.figure()

fig, axes = plt.subplots(1, 4, figsize=(15,15))

ax1 = axes[0]

ax1.imshow(sample_images_positive[0], cmap='Greys_r')

ax1.set_title('infected')

ax1.axis('off')

ax2 = axes[1]

ax2.imshow(sample_images_positive[5], cmap='Greys_r')

ax2.set_title('infected')

ax2.axis('off')

ax3 = axes[2]

ax3.imshow(sample_images_positive[9], cmap='Greys_r')

ax3.set_title('infected')

ax3.axis('off')

ax4 = axes[3]

ax4.imshow(sample_images_negative[0], cmap='Greys_r')

ax4.set_title('uninfected')

ax4.axis('off');
image_grayscale = color.rgb2gray(sample_images_positive[5])

plt.imshow(image_grayscale, cmap='Greys_r')
def padding(image, size):

    desired_size = size

    old_size = image.size

    ratio = float(desired_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    im = image.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new('RGB',(desired_size, desired_size))

    new_im.paste(image, ((desired_size-new_size[0])//2,

                    (desired_size-new_size[1])//2))

    return new_im
def brighten(image):

    max_ = np.max(image)

    image[image==0.0] = max_

    return image
img = sample_images_positive[5]

img = np.array(img)

img = color.rgb2gray(img)

img = brighten(img)

plt.imshow(img, cmap='Greys_r');
X_positive_bright = []

for filename in listdir('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/'):

    if 'Thumbs.db' not in filename:

        img = Image.open('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/' + filename)

        img = padding(img, 220)

        img = np.array(img)

        img = color.rgb2gray(img)

        img = brighten(img.reshape(220*220))

        img = transform.resize(img.reshape(220,220), (100,100))

        X_positive_bright.append(img)

X_positive_bright = np.array(X_positive_bright)
X_negative_bright = []

for filename in listdir('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/'):

    if 'Thumbs.db' not in filename:

        img = Image.open('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/' + filename)

        img = padding(img, 220)

        img = np.array(img)

        img = color.rgb2gray(img)

        img = brighten(img.reshape(220*220))

        img = transform.resize(img.reshape(220,220), (100,100))

        X_negative_bright.append(img)

X_negative_bright = np.array(X_negative_bright)
X_positive_bright = X_positive_bright.reshape(13779, 100*100)

X_negative_bright = X_negative_bright.reshape(13779, 100*100)
X_positive_bright.min(), X_positive_bright.max(), X_negative_bright.min(), X_negative_bright.max()
from sklearn.decomposition import PCA

X_full = np.row_stack((X_positive_bright, X_negative_bright))

pca_full_variance = PCA(n_components = 0.95) 

pca_full_variance.fit(X_full) 

pca_full_variance.components_.shape
fig, axes = plt.subplots(3, 8, figsize=(12, 6),

                         subplot_kw={'xticks':[], 'yticks':[]},

                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):

    ax.imshow(pca_full_variance.components_[i].reshape(100, 100), cmap='Greys_r')
X_reduced = pca_full_variance.transform(X_full)

X_reduced.shape
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.metrics import plot_confusion_matrix
y_positive = np.ones(X_positive_bright.shape[0])

y_negative = np.zeros(X_negative_bright.shape[0])

y = np.hstack((y_positive, y_negative))
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, stratify=y, train_size=0.5)
model_svc = SVC(C=70782, gamma=0.000977)

model_svc.fit(X_train, y_train)
pred = model_svc.predict(X_test)
accuracy_score(y_test, pred)
plot_confusion_matrix(model_svc, X_test, y_test,

                     cmap=plt.cm.Blues,

                                 normalize='true')

plt.show();