# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

def crop_image_from_gray(img,tol=9):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img



def circle_crop(img):

    """

    Create circular crop around image centre

    """

    #img = cv2.imread(img)

    img = crop_image_from_gray(img)



    height, width, depth = img.shape

    largest_side = np.max((height, width))

    img = cv2.resize(img, (largest_side, largest_side))



    height, width, depth = img.shape



    x = int(width / 2)

    y = int(height / 2)

    r = np.amin((x, y))



    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)



    return Pimage.fromarray(img)
!ls -l /kaggle/input/
!rm -rf  /kaggle/working/filter

!mkdir filter

#!rm -rf /kaggle/working/circle_crop_atos

#!ls -hs /kaggle/working/circle_crop_atos

#file_names=os.listdir("../input/severstal-steel-defect-detection/train_images")

#os.listdir("../input/aptos2019-blindness-detection/train_images")

#file_names

#!ls -l /kaggle/input/ext-data/external_train/

#os.remove('/kaggle/working/filter/8cb6b5b2f19c.png')



!ls -l /kaggle/working/filter
#!ls -l /kaggle/input/rsna-hemorrhage-png/meta/meta/
import pandas as pd

import os

from fastai import *

from fastai.vision import *

import pydicom as pic

path_meta = Path('../input/rsna-hemorrhage-png/meta/meta/')

path = Path('../input/rsna-intracranial-hemorrhage-detection/')

path_str='../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'

path_trn = path/'stage_1_train_images'

path_tst = path/'stage_1_test_images'



df_comb = pd.read_feather(path_meta/'comb.fth').set_index('SOPInstanceUID')

df_tst  = pd.read_feather(path_meta/'df_tst.fth').set_index('SOPInstanceUID')

df_samp = pd.read_feather(path_meta/'wgt_sample.fth').set_index('SOPInstanceUID')
import os

import glob

from joblib import Parallel, delayed

from PIL import Image as Pimage

import matplotlib.pyplot as plt

import zipfile

#import pylot as plt

in_dir = 'train_orig/'

out_dir = 'train_96/'

IMAGE_SIZE = 96



from PIL import Image, ImageChops

#JPEG_FILES = glob.glob(in_dir+'*.jpeg')

def convert(img_file,path):

    ext='../input/ext-data/external_train/'

    #print('img',img_file)

    if os.path.isfile(ext+img_file):

        im = Pimage.open(ext+img_file)

        #print(im.shape)

    else:

        

        im = Pimage.open(path+img_file)

    #im=circle_crop(np.asarray(im))

    #print(im.size)

    #plt.imshow(im)

    #im.thumbnail((512,512),Pimage.ANTIALIAS)

    #im = crop_image_from_gray(np.asarray(im))

    #im  =cv2.bilateralFilter(np.asarray(im) ,8,10,10  )

    #kernel = np.ones((7,7),np.float32)/30

    #im = cv2.filter2D(im,-1,kernel)

    im=Pimage.fromarray(im)

    im.resize((328,328),Pimage.ANTIALIAS).save('/kaggle/working/filter/' + img_file, 'PNG')

    z.write('/kaggle/working/filter/' + img_file )

    os.remove('/kaggle/working/filter/' + img_file )



#file_names=os.listdir("../input/aptos2019-blindness-detection/train_images")

#file_names.append(os.listdir('../input/ext-data/external_train/'))

#file_names= np.hstack(file_names ).tolist()

#z = zipfile.ZipFile("/kaggle/working/filter/filter.zip", "w")

#Parallel(n_jobs=1, verbose=10)(delayed(convert)(f) for f in file_names)
from skimage import exposure

def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img

def fix_pxrepr(dcm):

    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return dcm

    x = dcm.pixel_array + 1000

    px_mode = 4096

    #print('shape',x.shape,dcm.pixel_array.shape)

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000

    return dcm

def window_image(pixel_array, window_center, window_width, is_normalize=True):

    image_min = window_center - window_width // 2

    image_max = window_center + window_width // 2

    image = np.clip(pixel_array, image_min, image_max)



    if is_normalize:

        

        image =exposure.equalize_adapthist(image)

        #image=exposure.adjust_gamma(image, gamma=1.5)

    else:

        

        image=(image-image_min)/(image_max-image_min)

    return image

def open_dicom(fname):

    fname = str(fname)

    dcm = pic.dcmread(fname) #'../input/stage_2_train_images/'

    #print(dcm['WindowCenter'].value,fname)

    #print(dcm.pixel_array.shape)

    '''  

    centre= (dcm['WindowCenter'].value)

    width= (dcm['WindowWidth'].value)

    

    if type(centre) in [pic.multival.MultiValue, tuple]:

        centre=float(centre[0])

    else :

        centre=float(centre)

    

    if type(width) in [pic.multival.MultiValue, tuple]:

        width=int(width[0])

    else :

        width=int(width)

    

   '''  

        

    dcm=fix_pxrepr(dcm)

    img= dcm.pixel_array 

    #level = 40; window = 80

    #

    img = img * int(dcm.RescaleSlope) + int(dcm.RescaleIntercept)

    #img = np.clip(img, centre - width // 2, centre + width // 2)

    img = crop_image_from_gray(np.asarray(img))  

    brain       = window_image(img, 40,  80)

    subdural    = window_image(img, 80, 200)

    soft_tissue = window_image(img, 40, 380)

    #brain = crop_image_from_gray(np.asarray(brain)).astype(np.uint8) 

    #subdural = crop_image_from_gray(np.asarray(subdural)).astype(np.uint8) 

    #soft_tissue = crop_image_from_gray(np.asarray(soft_tissue)).astype(np.uint8)

    #img= np.stack((img,)*3, axis=-1)

     

    #img = PIL.Image.fromarray(img).convert('RGB')

    im = crop_image_from_gray(np.asarray(img))  

    im = np.dstack([brain,subdural,soft_tissue])

    #im = crop_image_from_gray(np.asarray(im))

     

    #p2,p98=np.percentile(im,(1,98))

    #img=exposure.rescale_intensity(im,(p2,p98))

    

    #img=np.moveaxis(img,2,0 )

    #print(img.shape)

                             

    #return Image(pil2tensor(np.asarray(im)/255, np.float32).float())

    return im*255
#file_names=os.listdir("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/")
#!mkdir /kaggle/working/filter

#!ls -hs  /kaggle/working/filter/ 
#df_samp.index
import os

import glob

from joblib import Parallel, delayed

from PIL import Image as Pimage

import matplotlib.pyplot as plt

import zipfile

#import pylot as plt

in_dir = 'train_orig/'

out_dir = 'train_96/'

IMAGE_SIZE = 96



from PIL import Image, ImageChops

#JPEG_FILES = glob.glob(in_dir+'*.jpeg')

def convert(img_file,path):

    ext='../input/ext-data/external_train/'

    #print('img',img_file)

    if os.path.isfile(ext+img_file):

        im = Pimage.open(ext+img_file+'.dcm')

        #print(im.shape)

    else:

        

        im = open_dicom(path+img_file+'.dcm')

        #Pimage.open(path+img_file)

    #im=circle_crop(np.asarray(im))

    #print(im.size)

    #plt.imshow(im)

    #im.thumbnail((512,512),Pimage.ANTIALIAS)

    #im = crop_image_from_gray(np.asarray(im))

    #im  =cv2.bilateralFilter(np.asarray(im) ,8,10,10  )

    #kernel = np.ones((7,7),np.float32)/30

    #im = cv2.filter2D(im,-1,kernel)

    #print(im.shape,im.min(),im.max())

    im=Pimage.fromarray(im.astype(np.uint8))

    im.resize((488,488),Pimage.ANTIALIAS).save('/kaggle/working/filter/' + img_file+'.jpg', 'JPEG')

    z.write('/kaggle/working/filter/' + img_file+'.jpg' )

    os.remove('/kaggle/working/filter/' +  img_file+'.jpg' )





#file_names.append(os.listdir('../input/ext-data/external_train/'))

#file_names= np.hstack(file_names ).tolist()

file_names=df_samp.index.values

z = zipfile.ZipFile("/kaggle/working/filter/filter_rsna.zip", "w")

Parallel(n_jobs=1, verbose=1)(delayed(convert)(f,path_str) for f in file_names)
#a=open_dicom(path_str+file_names[16:50][16]) 

#plt.hist(a.flatten(),256)

#a=open_dicom(path_str+'ID_1bb6286d2'+'.dcm') 

#plt.imshow(a)
a=open_dicom(path_str+file_names[16:50][16]+'.dcm') 

plt.imshow(a)
#file_names=os.listdir("../input/aptos2019-blindness-detection/train_images")

#file_names.append(os.listdir('../input/ext-data/external_train/'))

#len(np.hstack(file_names).tolist() )

#file_names[-1]

#file_names[0:1]



#a=open_dicom(path_str+file_names[16:50][14]) 

#im=Pimage.fromarray(a.astype(np.uint8) )

#plt.imshow(a)
#a=open_dicom(path_str+file_names[0:1][0]) 

#im=Pimage.fromarray(a.astype(np.uint8) )

#plt.imshow(a)
#im2 = Pimage.open('/kaggle/working/filter/'+'ID_1bb6286d2.jpg')



#plt.imshow(im2)
#file_names=os.listdir("../input/aptos-preprocessed-420x420/val-imgs")

#file_names[0]
#im = Pimage.open('/kaggle/working/circle_crop_atos/'+'a87f53bc984a.png')



#plt.imshow(im)
#im = Pimage.open('/kaggle/working/circle_crop_atos/'+'a87f53bc984a.png')



#plt.imshow(im)
def __butterworth_filter( I_shape, filter_params=[30,2]):

        P = I_shape[0]/2

        Q = I_shape[1]/2

        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')

        Duv = (((U-P)**2+(V-Q)**2)).astype(float)

        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])

        return (1 - H)
def __apply_filter( I, H):

        H = np.fft.fftshift(H)

        I_filtered = (0.75 + 1.25*H)*I

        return I_filtered

  
#import pandas as pd

#df_orig=pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import * 

from fastai.vision import *
#df_orig
'''

df_orig['ImageId']=df_orig.ImageId_ClassId.apply(lambda x : x.split('_')[0])

df_orig['class_id']=df_orig.ImageId_ClassId.apply(lambda x : x.split('_')[1])

#df_train=df_orig.loc[~df_train.EncodedPixels.isnull()]

df_train_final=pd.DataFrame({'ImageId':[]})

df_train_final['ImageId']=df_orig[~df_orig.EncodedPixels.isnull()].ImageId.unique() 

#df_train_final['ImageId']=df_train.ImageId.unique() 

'''
#df_orig.loc[df_orig.ImageId_ClassId=='0002cc93b.jpg_1',['EncodedPixels','class_id']]
'''

#df_orig

l_1=[]

l_2=[]

l_3=[]

l_4=[]



for i in df_train_final.ImageId:

    



#df_orig[df_orig.ImageId_Classid=='0002cc93b.jpg_1'].EncodedPixels.values

    for rle in (df_orig.loc[df_orig.ImageId==i,['EncodedPixels','class_id']].values):

        #print(rle[0])

        if isinstance(rle[0], str):

            

            mask=open_mask_rle( rle[0],(256,1600)).px.permute(0,2,1) 

        #print(mask.shape)

        #print(mask.sum())

            if rle[1]=='1':

                l_1.append(mask.sum())

            if rle[1]=='2':

                l_2.append(mask.sum())

            if rle[1]=='3':

                l_3.append(mask.sum())

            if rle[1]=='4':

                l_4.append(mask.sum())

'''
 #plt.hist(l_2)

    
'''

import time

img1 = cv2.imread('../input/severstal-steel-defect-detection/train_images/0007a71bf.jpg' )

a=time.time()

img = cv2.imread('../input/severstal-steel-defect-detection/train_images/fff02e9c5.jpg',0)

print(img.shape)

#img = np.float32(img)

#img = img/255

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

 

#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

I_log = np.log1p(np.array(img, dtype="float"))

I_fft = np.fft.fft2(I_log)





        # Filters



H = __butterworth_filter(I_shape = I_fft.shape, filter_params = [30,2])



I_fft_filt = __apply_filter(I = I_fft, H = H)

I_filt = np.fft.ifft2(I_fft_filt)

I = np.exp(np.real(I_filt))-1

I = cv2.cvtColor(np.uint8(I), cv2.COLOR_GRAY2RGB)

from skimage import exposure

 

p2,p98=np.percentile(I,(2,99))

I=exposure.rescale_intensity(I,(p2,p98))

b=time.time()

print(b-a)

'''
#plt.imshow(np.uint8(I))
#plt.imshow(img1)
'''

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import * 

from fastai.vision import *

open_mask_rle(df[df.ImageId_ClassId=='fff02e9c5.jpg_3'].EncodedPixels.values[0],(256,1600))

'''
'''

import cv2

import numpy as np

import matplotlib.pyplot as plt



img = cv2.imread('../input/severstal-steel-defect-detection/train_images/ebce68542.jpg',-1)

print(img.shape)

#img = np.float32(img)

#img = img/255

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img=cv2.bilateralFilter(img ,10,10,10  )

rows,cols,dim=img.shape



rh, rl, cutoff = 2.5,0.5,32



imgYCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

y,cr,cb = cv2.split(imgYCrCb)



y_log = np.log(y+0.01)



y_fft = np.fft.fft2(y_log)



y_fft_shift = np.fft.fftshift(y_fft)





DX = cols/cutoff

G = np.ones((rows,cols))

for i in range(rows):

    for j in range(cols):

        G[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*DX**2))))+rl



result_filter = G * y_fft_shift



result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))



result = np.exp(result_interm)

'''

#plt.imshow(imgYCrCb )

#im  =cv2.median

#cv2.bilateralFilter(np.asarray(img) ,10,10,10  )

#plt.imshow(im)
#plt.imshow(img)#