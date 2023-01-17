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
import numpy

import cv2



img = cv2.imread('../input/image3/download (2).jpg',1)

print(img)

from PIL import Image

import cv2

import numpy as np

from skimage import io

import os.path

from skimage.io import imread

from skimage import data_dir

import matplotlib.pyplot as plt

import numpy as np



print(img.shape)

print(type(img.shape))

img = cv2.imread('../input/image3/download (2).jpg',1)



def jpeg_res(img):



   # open image for reading in binary mode

    mg = cv2.imread('../input/image3/download (2).jpg',1)

    with open(img,'rb') as img_file:



       # height of image (in 2 bytes) is at 164th position

       img_file.seek(163)



       # read the 2 bytes

       a = img_file.read(2)



       # calculate height

       height = (a[0] << 8) + a[1]



       # next 2 bytes is width

       a = img_file.read(2)



       # calculate width

       width = (a[0] << 8) + a[1]



print("The resolution of the image is",width,"x",height)



jpeg_res(img)
import cv2

import numpy as np

from skimage import io

img=io.imread('../input/image3/download (2).jpg')

print(img)
import cv2

import numpy as np

from skimage import io

import os.path

from skimage.io import imread

from skimage import data_dir

import matplotlib.pyplot as plt

import numpy as np

img = cv2.imread('../input/image3/download (2).jpg')

print(img)
path1 = os.path.abspath('../input/dirct4')

path2 = os.path.abspath('../input/dirct5')

#path3 = os.path.abspath('Type_3')

folder = os.path.join(path1, path2)



def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):

        if filename.endswith(".jfif"):

            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:

                images.append(img)

            return images

        else:

            print("none")



print(load_images_from_folder(folder))

print(len((path1)))


from PIL import Image

 

img = cv2.imread('../input/image3/download (2).jpg')

width, height = img.size

print("The dimension of the image is", width, "x", height)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
import cv2

import numpy

img=cv2.imread('../input/image3/download (2).jpg')

avgc=numpy.average(img,axis=1)

avg_row=numpy.average(avgc,axis=1)

print(avg_row)

from os import listdir

from PIL import Image as PImage





def loadImages(path):

    # return array of images



    imagesList = listdir(path)

    loadedImages = []

    for image in imagesList:

        img = PImage.open(path + image)

        loadedImages.append(img)



    return loadedImages



path = imread('../input/dirct4')



# your images in an array

imgs = loadImages(path)
import glob

import cv2

import os

#path2 = os.path.abspath('../input/dirct5')

img_dir= os.path.abspath('../input/dirct5')

data_path = os.path.join(img_dir,'*g')

files= glob.glob(data_path)

data = []

for f1 in files:

    img = cv2.imread(f1)

    data.append(img)    

print(data)



import xlsxwriter

import numpy as np

import pandas

import glob

import cv2

import os

from xlwt import Workbook

#wb=Workbook()

#sheet=wb.add_sheet('sheet 1')

workbook = xlsxwriter.Workbook('../input/ABCDEF/nisha.xlsx')

worksheet = workbook.add_worksheet()

path2 = os.path.abspath('../input/dirct5')

img_dir= os.path.abspath('../input/dirct5')

data_path = os.path.join(img_dir,'*g')

files= glob.glob(data_path)

data = []

for f1 in files:

    img = cv2.imread(f1)

    data.append(img)

    

print(data)

wb=Workbook()

sheet=wb.add_sheet('sheet 1')

for i in range(len(data)):

    sheet.write(i,0,i)

wb.save('data2.xls')    



#row = 0



#for col, data1 in enumerate(data):

    #worksheet.write_column(row, col, data1)



#workbook.close()

from PIL import Image



filePath = '../input/image3/download (2).jpg'

img = Image.open(filePath)

print(img)

width, height = img.size

print("The dimension of the image is", width, "x", height)



import cv2

import numpy as np

#import os



file1= os.path.abspath('../input/dirct4')

file2= os.path.abspath('../input/dirct5')



for f1 in file1:

    image1 = cv2.imread(f1)

    



for f2 in file2:

    image2 = cv2.imread(f2)

difference = cv2.subtract(image1, image2)



result = not np.any(difference) #if difference is all zeros it will return False

print(result)



if result is True:

     print("The images are the same")

else:

     cv2.imwrite("result.jpg", difference)

     print ("the images are different")