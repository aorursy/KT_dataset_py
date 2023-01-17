import numpy as np 

import pandas as pd

import os

print(os.listdir("../input"))
import pandas as pd

df = pd.read_csv("../input/train.csv")

df.head()
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from PIL import Image

import matplotlib.patches as patches

import cv2

kal = np.random.randint(1337)

sal = str(df.loc[kal][1]).split()

print(df.loc[kal][0])

img = np.array(Image.open('../input/train_images/'+str(df.loc[kal][0])+'.jpg'), dtype=np.uint8)

fig,ax = plt.subplots(1)

#fig = plt.figure()

X = []

for i in range(0,len(sal),5):

    rect = patches.Rectangle((int(sal[i+1]),int(sal[i+2])),int(sal[i+3]),int(sal[i+4]),linewidth=1,edgecolor='r',facecolor='none')

    ax.add_patch(rect)

    X.append(img[int(sal[i+2]):int(sal[i+2])+int(sal[i+4]),int(sal[i+1]):int(sal[i+1])+int(sal[i+3]),:])

'''for i in range(0,len(X)):

    imgnew = Image.fromarray(X[0])

    fig.add_subplot(i,2,1)

    imgplot = plt.imshow(imgnew)

plt.show()

#imgi = plt.imshow(img)'''

#show_images(X)

xa = Image.fromarray(X[0])

edges = cv2.Canny(X[0],100,200)

plt.imshow(edges,cmap='gray')

#plt.imshow(img)

plt.show()

def show_images(images, cols = 1, titles = None):

    """Display a list of images in a single figure with matplotlib.

    

    Parameters

    ---------

    images: List of np.arrays compatible with plt.imshow.

    

    cols (Default = 1): Number of columns in figure (number of rows is 

                        set to np.ceil(n_images/float(cols))).

    

    titles: List of titles corresponding to each image. Must have

            the same length as titles.

    """

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)

        if image.ndim == 2:

            plt.gray()

        plt.imshow(image)

        a.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
