# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import sys

import numpy as np

import matplotlib

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import glob



from sklearn import decomposition

from sklearn.neighbors import KernelDensity

from sklearn.manifold import TSNE



matplotlib.style.use('fivethirtyeight')



#%% load the data, go over training images and store them in a list



dataDir = '../input/ClubPictures/'

faceFiles = glob.glob(dataDir + '*.png')



listOfPlayerNames = []

listOfImages = []

for imageFilename in faceFiles:

    currName = imageFilename.split("/")[-1].split('.')[0]

        

    try:

        currImage = mpimg.imread(imageFilename)

        listOfPlayerNames.append(currName)

        listOfImages.append(currImage)

    except:

        print("didn't load '" + currName + "'")
#%% show some images



matplotlib.rcParams['font.size'] = 9

matplotlib.rcParams['figure.figsize'] = (12,12)



numRows = 5; numCols = 5



plt.figure()

for k in range(numRows*numCols):

    randInd = np.random.randint(len(listOfImages))

    plt.subplot(numRows,numCols,k+1); 

    plt.imshow(listOfImages[randInd])

    plt.title(listOfPlayerNames[randInd]); plt.axis('off')
import cv2



base = '../input/ClubPictures/'

images = os.listdir(base)

output = cv2.imread(base+images[0])

image1 = cv2.imread(base+images[1])

div = 1.0/len(images)

cv2.addWeighted(image1, div, output, div, 0.3, output)



for i in range(2,len(images)):



	# load the image

	image1 = cv2.imread(base+images[i])

	cv2.addWeighted(image1, div, output, 1, 0.3, output)

cv2.imwrite("Output1.jpg", output)
from matplotlib.pyplot import imshow

from PIL import Image

from PIL import ImageFont

from PIL import ImageDraw 



%matplotlib inline

pil_im = Image.open('Output1.jpg', 'r')

draw = ImageDraw.Draw(pil_im)

draw.text((100, 200),"KAGGLE FC",(255,255,255))

imshow(np.asarray(pil_im))