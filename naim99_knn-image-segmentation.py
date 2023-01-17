from numpy import *

import numpy

numpy.version.version
import cv2

import numpy

import matplotlib.pyplot as plt

from scipy.cluster.vq import *

img = cv2.imread('../input/defected-brain-image/meningioma-care-scan.jpg')

plt.imshow(img)
center = 3

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

datalab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

column = len(datalab[1,])

rows = len(datalab)
column
rows
ab = datalab[:,:,2:3]

ab = reshape(ab,rows*column,order="C")

print(ab.shape)
ans ,arr = kmeans2(ab.astype(float),center,iter=15,missing='warn')
cluster = []

a = []

for i in range(center):

    cluster.append(a)

    a = []
arr = reshape(arr,(rows,column),order='C')
img_backup = img.copy()

print (ans)
for i in range(rows):

    for j in range(column):

        img_backup[i,j] = [0,0,0]
for z in range(center):

    for x in range(rows):

        for y in range(column):

            if arr[x,y] == z:

                #print z

                img_backup[x,y] = img[x,y]

                #cluster[z].append([x,y])

    cv2.imwrite('%s.jpg'%z,img_backup)

    print ('cluster%s'%z)
img_0 = cv2.imread('./0.jpg')

img_0 = img_0[:,:,0]

plt.imshow(img_0)
img_1 = cv2.imread('./1.jpg')

img_1 = img_1[:,:,0]

plt.imshow(img_1)
img_2 = cv2.imread('./2.jpg')

img_2 = img_2[:,:,0]

plt.imshow(img_2)
img_0.shape
import cv2

import numpy

from scipy.cluster.vq import *

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt

from scipy import ndimage as ndi



from skimage import data

from skimage.metrics import (adapted_rand_error,

                              variation_of_information)

from skimage.filters import sobel

from skimage.measure import label

from skimage.util import img_as_float

from skimage.feature import canny

from skimage.morphology import remove_small_objects

from skimage.segmentation import (morphological_geodesic_active_contour,

                                  inverse_gaussian_gradient,

                                  watershed,

                                  mark_boundaries)

 

import matplotlib.pyplot as plt

from skimage import data, img_as_float

from skimage.segmentation import chan_vese
#image = img_as_float(data.camera())

# Feel free to play around with the parameters to see how they impact the result

cv = chan_vese(img_0, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=40,

               dt=0.5, init_level_set="checkerboard", extended_output=True)



fig, axes = plt.subplots(2, 2, figsize=(8, 8))

ax = axes.flatten()



ax[0].imshow(img_0, cmap="gray")

ax[0].set_axis_off()

ax[0].set_title("Original Image", fontsize=12)



ax[1].imshow(cv[0], cmap="gray")

ax[1].set_axis_off()

title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))

ax[1].set_title(title, fontsize=12)



ax[2].imshow(cv[1], cmap="gray")

ax[2].set_axis_off()

ax[2].set_title("Final Level Set", fontsize=12)



ax[3].plot(cv[2])

ax[3].set_title("Evolution of energy over iterations", fontsize=12)



fig.tight_layout()

plt.show()
img_cv = cv[1]

plt.imshow(img_cv, cmap='gray')
elevation_map = sobel(img_cv)

markers = np.zeros_like(img_cv)

markers[img_0 < 30] = 1

markers[img_0 > 150] = 2

im_true = watershed(elevation_map, markers)

im_true = ndi.label(ndi.binary_fill_holes(im_true - 1))[0]

edges = sobel(img_cv)

im_test1 = watershed(edges, markers=468, compactness=0.001)

edges = canny(img_cv)

fill_coins = ndi.binary_fill_holes(edges)

im_test2 = ndi.label(remove_small_objects(fill_coins, 21))[0]



image = img_as_float(img_cv)

gradient = inverse_gaussian_gradient(image)

init_ls = np.zeros(image.shape, dtype=np.int8)

init_ls[10:-10, 10:-10] = 1

im_test3 = morphological_geodesic_active_contour(gradient, iterations=500,

                                                 init_level_set=init_ls,

                                                 smoothing=1, balloon=-1,

                                                 threshold=0.69)

im_test3 = label(im_test3)



method_names = ['Compact watershed', 'Canny filter',

                'Morphological Geodesic Active Contours']

short_method_names = ['Compact WS', 'Canny', 'GAC']



precision_list = []

recall_list = []

split_list = []

merge_list = []

for name, im_test in zip(method_names, [im_test1, im_test2, im_test3]):

    error, precision, recall = adapted_rand_error(im_true, im_test)

    splits, merges = variation_of_information(im_true, im_test)

    split_list.append(splits)

    merge_list.append(merges)

    precision_list.append(precision)

    recall_list.append(recall)

    print(f"\n## Method: {name}")

    print(f"Adapted Rand error: {error}")

    print(f"Adapted Rand precision: {precision}")

    print(f"Adapted Rand recall: {recall}")

    print(f"False Splits: {splits}")

    print(f"False Merges: {merges}")



fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)

ax = axes.ravel()



ax[0].scatter(merge_list, split_list)

for i, txt in enumerate(short_method_names):

    ax[0].annotate(txt, (merge_list[i], split_list[i]),

                   verticalalignment='center')

ax[0].set_xlabel('False Merges (bits)')

ax[0].set_ylabel('False Splits (bits)')

ax[0].set_title('Split Variation of Information')



ax[1].scatter(precision_list, recall_list)

for i, txt in enumerate(short_method_names):

    ax[1].annotate(txt, (precision_list[i], recall_list[i]),

                   verticalalignment='center')

ax[1].set_xlabel('Precision')

ax[1].set_ylabel('Recall')

ax[1].set_title('Adapted Rand precision vs. recall')

ax[1].set_xlim(0, 1)

ax[1].set_ylim(0, 1)



ax[2].imshow(mark_boundaries(image, im_true))

ax[2].set_title('True Segmentation')

ax[2].set_axis_off()



ax[3].imshow(mark_boundaries(image, im_test1))

ax[3].set_title('Compact Watershed')

ax[3].set_axis_off()



ax[4].imshow(mark_boundaries(image, im_test2))

ax[4].set_title('Edge Detection')

ax[4].set_axis_off()



ax[5].imshow(mark_boundaries(image, im_test3))

ax[5].set_title('Morphological GAC')

ax[5].set_axis_off()



plt.show()

pic = plt.imread('../input/skin-cancer/cccc.jpg')/255  # dividing by 255 to bring the pixel values between 0 and 1

print(pic.shape)

plt.imshow(pic)
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])

pic_n.shape
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(pic_n)

pic2show = kmeans.cluster_centers_[kmeans.labels_]
cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])

plt.imshow(cluster_pic)