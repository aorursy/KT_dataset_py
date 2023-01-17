# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pydicom as dicom

import numpy as np

from PIL import Image

import cv2

import matplotlib.pyplot as plt

import deepdish as dd

import numpy as np

from PIL import Image

import glob

import cv2

import os

import shutil

import tarfile


!mkdir /kaggle/first/



name = '/kaggle/input/part1xray/1/'  

a = len('/kaggle/input/part1xray/1/9352020/20130603/')

folder_name = os.listdir(name)



i=0

for folder in folder_name:

    try:

        jpg_name = glob.glob(name+folder+"/*/*.jpg")[0][a:]

    #     tar = glob.glob(name+folder+"/*/*.tar.gz")

    #     my_tar = tarfile.open(tar[0])

    #     my_tar.extractall(tar[0][:-16]) # specify which folder to extract to

    #     my_tar.close()



        dicom1 = glob.glob(name+folder+"/*/*/001")[0]

        data = dicom.read_file(dicom1)



        img = np.frombuffer(data.PixelData, dtype=np.uint16).copy()

        if data.PhotometricInterpretation == 'MONOCHROME1':

            img = img.max() - img

        img = img.reshape((data.Rows, data.Columns))

        img1 = (img/img.max())*256



        #img1 = Reshape_Image(img1)



        cv2.imwrite('/kaggle/first/'+jpg_name,img1) 

        i=i+1

        if i %100 ==0:

            print(i)



    except:

        pass

        print(name+folder)    

name = '/kaggle/input/part2xray/2/'  

a = len('/kaggle/input/part2xray/2/9352020/20130603/')

folder_name = os.listdir(name)



i=0

for folder in folder_name:

    try:

        jpg_name = glob.glob(name+folder+"/*/*.jpg")[0][a:]

    #     tar = glob.glob(name+folder+"/*/*.tar.gz")

    #     my_tar = tarfile.open(tar[0])

    #     my_tar.extractall(tar[0][:-16]) # specify which folder to extract to

    #     my_tar.close()



        dicom1 = glob.glob(name+folder+"/*/*/001")[0]

        data = dicom.read_file(dicom1)



        img = np.frombuffer(data.PixelData, dtype=np.uint16).copy()

        if data.PhotometricInterpretation == 'MONOCHROME1':

            img = img.max() - img

        img = img.reshape((data.Rows, data.Columns))

        img1 = (img/img.max())*256



        #img1 = Reshape_Image(img1)



        cv2.imwrite('/kaggle/first/'+jpg_name,img1) 

        i=i+1

        if i %100 ==0:

            print(i)



    except:

        pass

        print(name+folder)    
name = '/kaggle/input/hiparthoplasty6e1/6.E.1/'  

a = len('/kaggle/input/hiparthoplasty6e1/6.E.1/9352020/20130603/')

folder_name = os.listdir(name)



i=0

for folder in folder_name:

    try:

        jpg_name = glob.glob(name+folder+"/*/*.jpg")[0][a:]

    #     tar = glob.glob(name+folder+"/*/*.tar.gz")

    #     my_tar = tarfile.open(tar[0])

    #     my_tar.extractall(tar[0][:-16]) # specify which folder to extract to

    #     my_tar.close()



        dicom1 = glob.glob(name+folder+"/*/*/001")[0]

        data = dicom.read_file(dicom1)



        img = np.frombuffer(data.PixelData, dtype=np.uint16).copy()

        if data.PhotometricInterpretation == 'MONOCHROME1':

            img = img.max() - img

        img = img.reshape((data.Rows, data.Columns))

        img1 = (img/img.max())*256



        #img1 = Reshape_Image(img1)



        cv2.imwrite('/kaggle/first/'+jpg_name,img1) 

        i=i+1

        if i %100 ==0:

            print(i)



    except:

        pass

        print(name+folder)    

name = '/kaggle/input/hiparthoplasty6e2/6.E.2/'  

a = len('/kaggle/input/hiparthoplasty6e2/6.E.2/9352020/20130603/')

folder_name = os.listdir(name)



i=0

for folder in folder_name:

    try:

        jpg_name = glob.glob(name+folder+"/*/*.jpg")[0][a:]

    #     tar = glob.glob(name+folder+"/*/*.tar.gz")

    #     my_tar = tarfile.open(tar[0])

    #     my_tar.extractall(tar[0][:-16]) # specify which folder to extract to

    #     my_tar.close()



        dicom1 = glob.glob(name+folder+"/*/*/001")[0]

        data = dicom.read_file(dicom1)



        img = np.frombuffer(data.PixelData, dtype=np.uint16).copy()

        if data.PhotometricInterpretation == 'MONOCHROME1':

            img = img.max() - img

        img = img.reshape((data.Rows, data.Columns))

        img1 = (img/img.max())*256



        #img1 = Reshape_Image(img1)



        cv2.imwrite('/kaggle/first/'+jpg_name,img1) 

        i=i+1

        if i %100 ==0:

            print(i)



    except:

        pass

        print(name+folder)    

name = '/kaggle/input/hiparthoplasty6c2/6.C.2/'  

a = len('/kaggle/input/hiparthoplasty6c2/6.C.2/9352020/20130603/')

folder_name = os.listdir(name)



i=0

for folder in folder_name:

    try:

        jpg_name = glob.glob(name+folder+"/*/*.jpg")[0][a:]

    #     tar = glob.glob(name+folder+"/*/*.tar.gz")

    #     my_tar = tarfile.open(tar[0])

    #     my_tar.extractall(tar[0][:-16]) # specify which folder to extract to

    #     my_tar.close()



        dicom1 = glob.glob(name+folder+"/*/*/001")[0]

        data = dicom.read_file(dicom1)



        img = np.frombuffer(data.PixelData, dtype=np.uint16).copy()

        if data.PhotometricInterpretation == 'MONOCHROME1':

            img = img.max() - img

        img = img.reshape((data.Rows, data.Columns))

        img1 = (img/img.max())*256



        #img1 = Reshape_Image(img1)



        cv2.imwrite('/kaggle/first/'+jpg_name,img1) 

        i=i+1

        if i %100 ==0:

            print(i)



    except:

        pass

        print(name+folder)    

!ls /kaggle/first | wc -l 
from PIL import Image

Image.open('/kaggle/first/03093003_1x1.jpg')