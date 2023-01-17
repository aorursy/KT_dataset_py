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
data=pd.read_csv("/kaggle/input/siim-medical-images/overview.csv")

data.head()
data.isnull().sum()
data["Age"].median()
%matplotlib inline

import matplotlib.pyplot as plt

import pydicom

one_dicom=pydicom.read_file("/kaggle/input/siim-medical-images/dicom_dir/ID_0077_AGE_0074_CONTRAST_0_CT.dcm")

one_dicom
one_dicom.pixel_array.shape
plt.imshow(one_dicom.pixel_array)
plt.imshow(one_dicom.pixel_array,cmap="gray")
import skimage.io as imio

one_tiff=imio.imread("/kaggle/input/siim-medical-images/tiff_images/ID_0077_AGE_0074_CONTRAST_0_CT.tif")

plt.imshow(one_tiff,cmap="gray")
plt.imshow(one_dicom.pixel_array, cmap=plt.cm.bone)
plt.imshow(one_dicom.pixel_array, cmap=plt.cm.gist_heat)
plt.imshow(one_dicom.pixel_array, cmap=plt.cm.PuBuGn)
data.head()
dicom_pixels=[]

paths=[]

directory="/kaggle/input/siim-medical-images/dicom_dir/"

for dicom_path in data["dicom_name"]:

    dicom=pydicom.read_file(directory+dicom_path)

    dicom_pixels.append(np.array(dicom.pixel_array))

    

dicom_pixels
len(dicom_pixels)
dataset=pd.DataFrame({

    "Age":data["Age"],

    "images":dicom_pixels

})
from fastai.vision import *

data = (ImageList.from_df(data,'/kaggle/input/siim-medical-images/tiff_images/',cols="tiff_name")

        .split_none()

        .label_from_df(cols="Age")

        .transform(get_transforms(), size=32)

        .databunch()

        .normalize(imagenet_stats))
imio.imsave(fname="dicomdeneme.jpg",arr=dataset["images"][0])
resim=imio.imread(fname="dicomdeneme.jpg")

plt.imshow(resim,cmap="gray")
plt.imshow(dataset["images"][0],cmap="gray")
plt.imsave(fname="dicomdeneme2.jpg",arr=dataset["images"][0])

resim=imio.imread(fname="dicomdeneme2.jpg")

plt.imshow(resim,cmap=plt.cm.gray)
plt.imshow(resim[:,:,1],cmap=plt.cm.gray)
plt.imsave(fname="dicomdeneme2.jpg",arr=dataset["images"][0],cmap="gray")

resim=imio.imread(fname="dicomdeneme2.jpg")

plt.imshow(resim,cmap=plt.cm.gray)
resim.shape
sirdominicpointer=0

path="images/"

! mkdir images

for img in dataset["images"]:

    plt.imsave(fname=path+"dicom_to"+str(sirdominicpointer)+".jpg",arr=dataset["images"][0],cmap="gray")

    sirdominicpointer+=1

    
sirdominicpointer=0

path="images2/train/"

! mkdir images2

!mkdir images2/train

for img in dataset["images"]:

    plt.imsave(fname=path+"dicom_to"+str(sirdominicpointer)+".jpg",arr=dataset["images"][0],cmap="gray")

    sirdominicpointer+=1
from fastai.vision.gan import *
generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=2)

critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=2)

ganlist=GANItemList.from_folder(path="images2")



databun=ganlist.split_none().label_from_func(noop).transform(tfms=[[crop_pad(size=64, row_pct=(0,1), col_pct=(0,1))], []], size=64, tfm_y=True).databunch()

learn = GANLearner.wgan(databun, generator, critic, switch_eval=False,opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)

learn.fit(88)

learn.gan_trainer.switch(gen_mode=True)

learn.show_results(ds_type=DatasetType.Train, rows=16, figsize=(14,14))
! rm -rf images2/