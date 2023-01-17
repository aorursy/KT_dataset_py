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
        break
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
trainpath = '/kaggle/input/siim-isic-melanoma-classification/train.csv'
traindf = pd.read_csv(trainpath)
traindf.head()
traindf['age_approx'].describe()
JPEG_PATH = '/kaggle/input/siim-isic-melanoma-classification/jpeg/'
DICOM_PATH = '/kaggle/input/siim-isic-melanoma-classification/'
traindf['target'].plot(kind='hist')
plt.xlabel("Category")
plt.ylabel("Images")
traindf['target'].value_counts()
traindf['benign_malignant'].value_counts()
malignDF = traindf[traindf['target'] == 1]
malignDF.head()
import pydicom as dicom
import imageio
import matplotlib.pylab as plt


malignImagPath = os.path.join(DICOM_PATH+'train/', 'ISIC_0149568.dcm')
malignImagPath2 = os.path.join(DICOM_PATH+'train/', 'ISIC_0188432.dcm')
malignImagPath3 = os.path.join(DICOM_PATH+'train/', 'ISIC_0207268.dcm')
malignImagPath4 = os.path.join(DICOM_PATH+'train/', 'ISIC_0247330.dcm')

ds = dicom.dcmread(malignImagPath)
ds2 = dicom.dcmread(malignImagPath2)
ds3 = dicom.dcmread(malignImagPath3)
ds4 = dicom.dcmread(malignImagPath4)



benignImagPath = os.path.join(DICOM_PATH+'train/', 'ISIC_2637011.dcm')
benignImagPath2 = os.path.join(DICOM_PATH+'train/', 'ISIC_0015719.dcm')
benignImagPath3 = os.path.join(DICOM_PATH+'train/', 'ISIC_0068279.dcm')
benignImagPath4 = os.path.join(DICOM_PATH+'train/', 'ISIC_0074268.dcm')

bs = dicom.dcmread(benignImagPath)
bs2 = dicom.dcmread(benignImagPath2)
bs3 = dicom.dcmread(benignImagPath3)
bs4 = dicom.dcmread(benignImagPath4)

f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(bs.pixel_array)
plot2.imshow(bs2.pixel_array)


f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(bs3.pixel_array)
plot2.imshow(bs4.pixel_array)
plt.title('Normal Images')
plt.show()

f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(ds.pixel_array)
plot2.imshow(ds2.pixel_array)


f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(ds3.pixel_array)
plot2.imshow(ds4.pixel_array)
plt.title('AbNormal Images')
plt.show()
ds.pixel_array.shape
ds.pixel_array.min()
ds.pixel_array.max()
from pydicom.pixel_data_handlers.util import convert_color_space 
rgbNormal = convert_color_space(bs.pixel_array, 'YBR_FULL_422', 'RGB')
rgbNormal2 = convert_color_space(bs2.pixel_array, 'YBR_FULL_422', 'RGB')
rgbNormal3 = convert_color_space(bs3.pixel_array, 'YBR_FULL_422', 'RGB')
rgbNormal4 = convert_color_space(bs4.pixel_array, 'YBR_FULL_422', 'RGB')
f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(rgbNormal)
plot2.imshow(rgbNormal2)


f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(rgbNormal3)
plot2.imshow(rgbNormal4)
plt.title('Normal Images')
plt.show()
rgbAbNormal = convert_color_space(ds.pixel_array, 'YBR_FULL_422', 'RGB')
rgbAbNormal2 = convert_color_space(ds2.pixel_array, 'YBR_FULL_422', 'RGB')
rgbAbNormal3 = convert_color_space(ds3.pixel_array, 'YBR_FULL_422', 'RGB')
rgbAbNormal4 = convert_color_space(ds4.pixel_array, 'YBR_FULL_422', 'RGB')
f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(rgbAbNormal)
plot2.imshow(rgbAbNormal2)


f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(rgbAbNormal3)
plot2.imshow(rgbAbNormal4)
plt.title('AbNormal Images')
plt.show()
orig_normal = [bs.pixel_array, bs2.pixel_array, bs3.pixel_array, bs4.pixel_array]
orig_abnormal = [ds.pixel_array, ds2.pixel_array, ds3.pixel_array, ds4.pixel_array]
rgbnormal_list = [rgbNormal, rgbNormal2, rgbNormal3, rgbNormal4]
rgbabnormal_list = [rgbAbNormal, rgbAbNormal2, rgbAbNormal3, rgbAbNormal4]
gray_normal = list(map(lambda image: cv2.cvtColor(image,cv2.COLOR_RGB2GRAY), rgbnormal_list))
gray_abnormal = list(map(lambda image: cv2.cvtColor(image,cv2.COLOR_RGB2GRAY), rgbabnormal_list))

f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(gray_normal[0])
plot2.imshow(gray_normal[1])

f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(gray_normal[2])
plot2.imshow(gray_normal[3])
plt.title('Normal Images')
f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(gray_abnormal[0])
plot2.imshow(gray_abnormal[1])

f, (plot1, plot2 ) = plt.subplots(1, 2)
plot1.imshow(gray_abnormal[2])
plot2.imshow(gray_abnormal[3])
plt.title('AbNormal Images')
import scipy.ndimage as ndi

columns = 3
# rows = len(os.listdir(DATA)) - 10
rows = 4
for i in range(len(rgbnormal_list)):    
    #histograms
    hist_original = ndi.histogram(orig_normal[i], min=0, max=255, bins=256)
    hist_rgb = ndi.histogram(rgbnormal_list[i], min=0, max=255, bins=256)
    hist_gray = ndi.histogram(gray_normal[i], min=0, max=255, bins=256)
    
    
    fig=plt.figure(figsize=(16, 16))
    fig.add_subplot(rows, columns, i+1)    
    plt.plot(hist_original)
    
    fig.add_subplot(rows, columns, i+1)   
    plt.plot(hist_rgb)
    
    fig.add_subplot(rows, columns, i+1)  
    plt.plot(hist_gray)
    plt.title("Normal "+str(i))

plt.show()

columns = 3
# rows = len(os.listdir(DATA)) - 10
rows = 4
for i in range(len(rgbabnormal_list)):    
    #histograms
    hist_original = ndi.histogram(orig_abnormal[i], min=0, max=255, bins=256)
    hist_rgb = ndi.histogram(rgbabnormal_list[i], min=0, max=255, bins=256)
    hist_gray = ndi.histogram(gray_abnormal[i], min=0, max=255, bins=256)
    
    
    fig=plt.figure(figsize=(16, 16))
    fig.add_subplot(rows, columns, i+1)    
    plt.plot(hist_original)
    
    fig.add_subplot(rows, columns, i+1)   
    plt.plot(hist_rgb)
    
    fig.add_subplot(rows, columns, i+1)  
    plt.plot(hist_gray)
    plt.title("AbNormal "+str(i))

plt.show()

normal = [bs.pixel_array, bs2.pixel_array, bs3.pixel_array, bs4.pixel_array]
abnormal = [ds.pixel_array, ds2.pixel_array, ds3.pixel_array, ds4.pixel_array]
normal_hists = [histNormal, histNormal2, histNormal3, histNormal4]
abnormal_hists = [histAbNormal, histAbNormal2, histAbNormal3, histAbNormal4]
for i in range(len(normal)):
    im = normal[i]
    hist = normal_hists[i]
    cdf = hist.cumsum() / hist.sum()
    im_equalized = cdf[im] * 255
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(im)
    axes[1].imshow(im_equalized)
    plt.show()
for i in range(len(abnormal)):
    im = abnormal[i]
    hist = abnormal_hists[i]
    cdf = hist.cumsum() / hist.sum()
    im_equalized = cdf[im] * 255
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(im)
    axes[1].imshow(im_equalized)
    plt.show()
