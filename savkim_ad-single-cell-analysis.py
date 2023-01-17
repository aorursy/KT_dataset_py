# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from PIL import Image

import numpy as np

import sys

import os

import csv



import cv2

import random

import matplotlib.pylab as plt

from glob import glob

import pandas as pd

from sklearn.model_selection import train_test_split
images = glob(os.path.join('../input', "*.png"))

# labels = pd.read_csv('../input/sample_labels.csv')



images[0:5]

r = random.sample(images, 3)



# Matplotlib black magic

plt.figure(figsize=(16,16))

plt.subplot(131)

plt.imshow(cv2.imread(r[0]))



plt.subplot(132)

plt.imshow(cv2.imread(r[1]))



plt.subplot(133)

plt.imshow(cv2.imread(r[2]));  
def createFileList(myDir, format='.png'):

    fileList = []

    print(myDir)

    for root, dirs, files in os.walk(myDir, topdown=False):

        for name in files:

            if name.endswith(format):

                fullName = os.path.join(root, name)

                fileList.append(fullName)

    return fileList



myFileList = createFileList('../input')





for file in myFileList:

    print(file)

    img_file = Image.open(file)

#     img_file.show()

    

    im_res = img_file.resize([28,28])

    pix = np.array(im_res)



    print(pix.size)

    width, height = pix.shape

   

    format = im_res.format

    mode = im_res.mode



    # Make image Greyscale

    img_grey = im_res.convert('L')

    #img_grey.save('result.png')

    #img_grey.show()



    # Save Greyscale values

    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))

    value = value.flatten()

    print(value)

    with open("img_pixels.csv", 'a') as f:

        writer = csv.writer(f, delimiter=',', quotechar='"',quoting=csv.QUOTE_ALL, lineterminator='\n')

        writer.writerow(value)
