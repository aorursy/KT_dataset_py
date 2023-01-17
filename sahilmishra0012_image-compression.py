import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
img=mpimg.imread('/kaggle/input/image.jpg')
imgplot = plt.imshow(img)
img=img/255
img.shape
img_new=np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
m,n=img_new.shape
k=2 #Number of colors we are choosing out of the image to be compressed
centroids=np.zeros((k,n))
# random initialization of Centroids.  

for i in range(k): 

    centroids[i] = img_new[int(np.random.random(1)*1000)]
def distance(x1,y1,z1,x2,y2,z2): 

    dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)+np.square(z1 - z2))

    return dist 
def nearest_centroid(image,centroid):

    centroids_dictionary={}

    for i in range(image.shape[0]):

        dlist=[]

        for j in range(k):

            dlist.append(distance(image[i][0],image[i][1],image[i][2],centroid[j][0],centroid[j][1],centroid[j][2]))

        min_dist=np.argmin(dlist)

        centroids_dictionary[i]=min_dist   

    return centroids_dictionary
def optimize(image,centroid,iterations):

    centroid_dictionary = nearest_centroid(image,centroid)

    for lo in tqdm(range(iterations)):

        print('Epoch'+str(lo))

        for key,value in centroid_dictionary.items():

            s=np.zeros((3,))

            count=0

            for i in range(img_new.shape[0]):

                if centroid_dictionary[i]==value:

                    s=s+img_new[i]

                    count=count+1

            s=s/count

            centroid[value]=s

        centroid_dictionary = nearest_centroid(image,centroid)

    return centroid_dictionary,centroid
%%time

centroids_dictionary,centroids=optimize(img_new,centroids,3)
img_compressed=np.zeros(img_new.shape)
for key, value in centroids_dictionary.items():

    img_compressed[key]=centroids[value]
img_compressed.shape
img_final=img_compressed.reshape((135, 165, 3))
img_final.shape
imgplot = plt.imshow(img_final)
mpimg.imsave('new_image.jpg',img_final)