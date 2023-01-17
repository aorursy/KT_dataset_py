# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import cv2 as cv2



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Install FAST AI lib

!pip install fastai2 -q
!pip install opencv-contrib-python
from fastai2.basics           import *

from fastai2.medical.imaging  import *

import matplotlib.pyplot as plt

import cv2
# It is just to try some code to check 

# if fast ai is fast enough

fn = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00026637202179561894768')

fname = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/13.dcm')

dcom = fname.dcmread()

dcom.show(scale=dicom_windows.lungs)
mask = dcom.mask_from_blur(dicom_windows.lungs)

wind = dcom.windowed(*dicom_windows.lungs)



_,ax = subplots(1,1)

show_image(wind, ax=ax[0])

show_image(mask, alpha=0.5, cmap=plt.cm.Reds, ax=ax[0]);
bbs = mask2bbox(mask)

lo,hi = bbs

show_image(wind[lo[0]:hi[0],lo[1]:hi[1]]);
path = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')

def fix_pxrepr(dcm):

    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000
#

# Here to read iamges one by one 

# Modify it to call in loop for all directories and images in it

# I have put only to call with one directory



def dcm_img(fn): 

    #fn = (path/fn).with_suffix('.dcm')

    fn = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/'+fn+'.dcm')

    try:

        x = fn.dcmread()

        fix_pxrepr(x)

    except Exception as e:

        pass

    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))

    

    

    return x

    #return 

    #return TensorImage(px.to_3chan(dicom_windows.lungs, bins=None))
# Gauusing blur

dcom_img = dcm_img(str(1))

gdcm = gauss_blur2d(dcom_img.windowed(*dicom_windows.lungs), 80) # using the brain for visualization purposes



_,ax = subplots(1,2)

show_image(gdcm,ax[0]);

show_image(dcom_img.windowed(*dicom_windows.lungs),ax[1]);
px = dcom_img.pixels.flatten()

plt.hist(px, bins=50, color='c');
# get image pixels directly from dcm image

tensor_dicom = dcom_img.hist_scaled()

tensor_dicom
#_,ax = subplots(1,2)

plt.imshow(tensor_dicom)
gdcm.save_jpg(path='test1.jpg' , wins=[dicom_windows.lungs,dicom_windows.lungs])

dcom_img.save_jpg(path='test2.jpg' , wins=[dicom_windows.lungs,dicom_windows.lungs])
# Load TIF image

from PIL import Image 

img1 = Image.open('test1.jpg')

img2 = Image.open('test2.jpg')



#img3 = cv2.addWeighted ( img1,4, img2 ,-4 ,128)

#plt.imshow(img3,cmap=plt.cm.bone)
#print(gdcm.dtype)

Tensor.save_tif16(gdcm,'test.tif')

#print(gdcm)
# Load TIF image

from PIL import Image 

tf_file = Image.open('test.tif')

print(tf_file.shape)

plt.imshow(tf_file,cmap=plt.cm.bone)
# Same image without Gaussian blur

# As it is very visible that there is lot of extra

# information which may not be good



dcom_img1 = dcm_img(str(5))

#gdcm = gauss_blur2d(dcom_img.windowed(*dicom_windows.lungs), 75) # using the brain for visualization purposes

show_image(dcom_img1.windowed(*dicom_windows.lungs));
dcom_img = dcm_img(str(13))

gdcm = gauss_blur2d(dcom_img.windowed(*dicom_windows.lungs), 75) # using the brain for visualization purposes





_,ax = subplots(1,2)

show_image(gdcm,ax[0]);

show_image(dcom_img.windowed(*dicom_windows.lungs),ax[1]);
# read new image

dcom_img = dcm_img(str(9))



mask = dcom_img.mask_from_blur(dicom_windows.lungs, sigma=0.1, thresh=0.75, remove_max=False)

#wind = dcom_img.windowed(*dicom_windows.lungs)

wind = gauss_blur2d(dcom_img.windowed(*dicom_windows.lungs), 25)



_,ax = subplots(1,2)

show_image(wind, ax=ax[0])

show_image(mask, alpha=0.5, cmap=plt.cm.Reds, ax=ax[1]);
bbs = mask2bbox(mask)

lo,hi = bbs

show_image(wind[lo[0]:hi[0],lo[1]:hi[1]]);
# Crop image using mask

mask_img =  wind[lo[0]:hi[0],lo[1]:hi[1]]



# convert into numpy if to use in TensorFlow

mask_img_array = mask_img.numpy()

#mask_img_array.shape



# lets see

plt.imshow(mask_img_array,cmap=plt.cm.bone)
import pydicom



patient_dir = '../input/osic-pulmonary-fibrosis-progression/train/ID00032637202181710233084'

#patient_dir = '../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430'

datasets = []



# First Order the files in the dataset

files = []

for dcm in list(os.listdir(patient_dir)):

    files.append(dcm) 

files.sort(key=lambda f: int(re.sub('\D', '', f)))



# Read in the Dataset

for dcm in files:

    path = patient_dir + "/" + dcm

    datasets.append(pydicom.dcmread(path))



# Plot the images

fig=plt.figure(figsize=(16, 6))

columns = 10

rows = 5



for i in range(1, columns*rows +1):

    img = datasets[i-1].pixel_array

    fig.add_subplot(rows, columns, i)

    plt.imshow(img, cmap="plasma")

    plt.title(i, fontsize = 9)

    plt.axis('off');
# image_path should have dir and image id

# e.g. - ID00007637202177411956430/1



def dcm_img_dir(image_path): 



    #fn = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/'+fn+'.dcm')

    fn = Path('../input/osic-pulmonary-fibrosis-progression/train/'+image_path+'.dcm')

    try:

        x = fn.dcmread()

        fix_pxrepr(x)

    except Exception as e:

        pass

    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))

    

    

    return x
def get_img_array(dcom_img):

         

    mask = dcom_img.mask_from_blur(dicom_windows.lungs, sigma=0.1, thresh=0.75, remove_max=False)

    #wind = dcom_img.windowed(*dicom_windows.lungs)

    wind = gauss_blur2d(dcom_img.windowed(*dicom_windows.lungs), 25)



    bbs = mask2bbox(mask)

    lo,hi = bbs



    # Crop image using mask

    mask_img =  wind[lo[0]:hi[0],lo[1]:hi[1]]



    # convert into numpy if to use in TensorFlow

    mask_img_array = mask_img.numpy()

    

    return mask_img_array

    
img_dir = 'ID00007637202177411956430/'

img_path = img_dir + '1'



dcom_img = dcm_img_dir(img_path)

img_array_1 = get_img_array(dcom_img)

img_array_1 = cv2.resize(img_array_1, (208, 511))
from skimage.transform import rescale, resize
img_dir = 'ID00007637202177411956430/'

img_path = img_dir + '13'



dcom_img = dcm_img_dir(img_path)

img_array_13 = get_img_array(dcom_img)

img_array_13_resize = resize(img_array_13, (208, 511))
plt.imshow(img_array_13,cmap=plt.cm.bone)
plt.imshow(img_array_13_resize,cmap=plt.cm.bone)
print(img_array_1.shape)

print(img_array_13.shape)
# lets see

#plt.imshow(img_array,cmap=plt.cm.bone)

import cv2 
sift = cv2.xfeatures2d.SIFT_create()



kp_1, desc_1 = sift.detectAndCompute(img_array_1, None)



kp_2, desc_2 = sift.detectAndCompute(img_array_13, None)



index_params = dict(algorithm=0, trees=5) 

search_params = dict() 

flann = cv2.FlannBasedMatcher(index_params, search_params)



matches = flann.knnMatch(desc_1, desc_2, k=2)



good_points = [] 

ratio = 0.3 



for m, n in matches: 

    if m.distance < ratio*n.distance: 

        good_points.append(m) 



print(len(good_points))