# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from IPython.display import Image

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

import cv2
filepath ="../input/greyscale-image/Image.jpg"
Image(filepath)
img = cv2.imread(filepath)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(img)

r = r.flatten()

g = g.flatten()

b = b.flatten()
fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(r, g, b)

plt.show()
vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)

print(vectorized.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K=3

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)

label = label.flatten()
center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))
plt.imshow(result_image)

plt.show()
figure_size = 10

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1, 2, 1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()
K=4

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)

label = label.flatten()

## Reshaping 2D array to 3D array

center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))
figure_size = 10

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1, 2, 1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()
K=5

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)

label = label.flatten()

    

## Reshaping 2D array to 3D array



center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))
figure_size = 10

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1, 2, 1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()
edges = cv2.Canny(img,150,200)

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray')

plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
pic = np.float64(img/256)

nsamples, nx, ny = pic.shape

pic1 = pic.reshape((nsamples,nx*ny))

kmeans = KMeans(n_clusters=20, random_state=0).fit(pic1)

pic2show = kmeans.cluster_centers_[kmeans.labels_]

plt.imshow(pic2show)
image = cv2.imread(filepath)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = image.reshape((image.shape[0] * image.shape[1], 3))
# cluster the pixel intensities

clt = KMeans(n_clusters = 5)

clt.fit(image)
def centroid_histogram(clt):

	# grab the number of different clusters and create a histogram

	# based on the number of pixels assigned to each cluster

	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one

	hist = hist.astype("float")

	hist /= hist.sum()

	# return the histogram

	return hist
def plot_colors(hist, centroids):

	# initialize the bar chart representing the relative frequency

	# of each of the colors

	bar = np.zeros((50, 300, 3), dtype = "uint8")

	startX = 0

	# loop over the percentage of each cluster and the color of

	# each cluster

	for (percent, color) in zip(hist, centroids):

		# plot the relative percentage of each cluster

		endX = startX + (percent * 300)

		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),

			color.astype("uint8").tolist(), -1)

		startX = endX

	

	# return the bar chart

	return bar
# build a histogram of clusters and then create a figure

# representing the number of pixels labeled to each color

hist = centroid_histogram(clt)

bar = plot_colors(hist, clt.cluster_centers_)

# show our color bart

plt.figure()

plt.axis("off")

plt.imshow(bar)

plt.show()