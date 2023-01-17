# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import shutil

os.makedirs('/kaggle/working/outputs')
# os.makedirs('../outputs')
ham1 =  os.listdir("/kaggle/input/skin-cancer-mnist-ham10000/ham10000_images_part_1")

ham2 =  os.listdir("/kaggle/input/skin-cancer-mnist-ham10000/ham10000_images_part_2")
ham1[0]
for i in range(len(ham1)):

    fname = ham1[i]

    src = os.path.join('/kaggle/input/skin-cancer-mnist-ham10000/ham10000_images_part_1', fname)

    # destination path to image

    dst = os.path.join('/kaggle/working/outputs',fname)

    # copy the image from the source to the destination

    shutil.copyfile(src, dst)
for i in range(len(ham2)):

    fname = ham2[i]

    src = os.path.join('/kaggle/input/skin-cancer-mnist-ham10000/ham10000_images_part_2', fname)

    # destination path to image

    dst = os.path.join('/kaggle/working/outputs',fname)

    # copy the image from the source to the destination

    shutil.copyfile(src, dst)
# !wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip
# import zipfile

# zip_ref = zipfile.ZipFile("ISIC2018_Task3_Test_Input.zip", 'r')

# zip_ref.extractall()

# zip_ref.close()



len(os.listdir('../input/isic2018-testset/ISIC2018_Task3_Test_Input'))
import glob

os.chdir('../input/isic2018-testset/ISIC2018_Task3_Test_Input')

aa = (glob.glob('*.jpg'))
aa[0]
for i in range(len(aa)):

    fname = aa[i]

    src = os.path.join('../ISIC2018_Task3_Test_Input', fname)

    # destination path to image

    dst = os.path.join('/kaggle/working/outputs',fname)

    # copy the image from the source to the destination

    shutil.copyfile(src, dst)
bb = os.listdir('/kaggle/working/outputs')

len(bb)
import cv2

import numpy as np

import matplotlib.pyplot as plt

i = 3

kernel = cv2.getStructuringElement(1,(17,17)) # Kernel for the morphological filtering






for i in range(50):

  idx=np.random.randint(1,len(bb))

  src = cv2.imread(os.path.join('/kaggle/working/outputs', bb[idx]))

  #print(src.shape)

  grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY ) #1 Convert the original image to grayscale

  blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) #2 Perform the blackHat filtering on the grayscale image to find the hair countours

  ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY) # intensify the hair countours in preparation for the inpainting algorithm

  dst = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA) # inpaint the original image depending on the mask



  #cv2.imwrite('thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])



  plt.figure(figsize=(20,10))

  plt.subplot(1,5,1).set_title('Original')

  plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), interpolation='nearest')

  plt.subplot(1,5,3).set_title('Grayscale')

  plt.imshow(cv2.cvtColor(grayScale, cv2.COLOR_BGR2RGB), interpolation='nearest')

  plt.subplot(1,5,4).set_title('Blackhat')

  plt.imshow(cv2.cvtColor(blackhat, cv2.COLOR_BGR2RGB), interpolation='nearest')

  plt.subplot(1,5,5).set_title('Thresh2')

  plt.imshow(cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB), interpolation='nearest')

  plt.subplot(1,5,2).set_title('Final')

  plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), interpolation='nearest')
import cv2

for i in range(len(bb)):

    if ((i+1)%10 == 0):

        print(i)

    idx=i

    src = cv2.imread(os.path.join('/kaggle/working/outputs', bb[idx]))

    #print(src.shape)

    grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY ) #1 Convert the original image to grayscale

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) #2 Perform the blackHat filtering on the grayscale image to find the hair countours

    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY) # intensify the hair countours in preparation for the inpainting algorithm

    dst = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA) # inpaint the original image depending on the mask

    cv2.imwrite('/kaggle/working/outputs'+ bb[i], dst)

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), interpolation='nearest')