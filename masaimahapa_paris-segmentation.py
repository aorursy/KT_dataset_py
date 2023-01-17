# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.utils.mem import *



! python -m pip install cityscapesscripts
val_dir='/kaggle/input/cityscapes-image-pairs/cityscapes_data/val/'

train_dir= '/kaggle/input/cityscapes-image-pairs/cityscapes_data/train/'
! mkdir /kaggle/working/train

!mkdir /kaggle/working/val



! mkdir /kaggle/working/train/train_masks

! mkdir /kaggle/working/train/train_images



! mkdir /kaggle/working/val/val_images

! mkdir /kaggle/working/val/val_masks
import matplotlib.pyplot as plt



img_dir=train_dir+'2345.jpg'

img= plt.imread(img_dir)

plt.imshow(img)

plt.show()
train_images_path='/kaggle/working/train/train_images/'

train_masks_path='/kaggle/working/train/train_masks/'



val_images_path='/kaggle/working/val/val_images/'

val_masks_path='/kaggle/working/val/val_masks/'
string = "Howdy doody"

string[::-1]
def divide_image_mask(image_path):

    try:

        image= cv2.imread(image_path)

        image=cv2.normalize(image,None,0,1 ,cv2.NORM_MINMAX,cv2.CV_32F)

        image=image[:,:,::1]

    

        

        picture= image[:,:256]

        mask=image[:,256:]

        

        

        

        return picture, mask

    except:

        print('dont exist')





for each in os.listdir(train_dir):



    picture, mask=divide_image_mask(train_dir+each)

    

    #training images

    os.chdir(train_images_path)

    cv2.imwrite(each, 255*picture) 

    

    #training masks

    mask_name= each.split('.')[0]

    mask_name=mask_name+'_P.jpg' 

        

    os.chdir(train_masks_path)

    cv2.imwrite(mask_name, 255*mask)

    
for each in os.listdir(val_dir):

    try:

        picture, mask=divide_image_mask(val_dir+each)

    

        #training images

        os.chdir(val_images_path)

        cv2.imwrite(each, 255*picture) 



        #training masks

        mask_name= each.split('.')[0]

        mask_name=mask_name+'_P.jpg' 



        os.chdir(val_masks_path)

        cv2.imwrite(mask_name, 255*mask)

    except:

        print('no such')

    
open_image(val_images_path+'10.jpg')

mask=open_mask(val_masks_path+'10_P.jpg')

mask
src_size = np.array(mask.shape[1:])

src_size
import cityscapesscripts as cs

from cityscapesscripts.helpers.labels import labels



id2name      = { name.id      : name for name in labels           }

id2name;

codes=[name.name for name in labels]

codes_dict={idx:name for idx, name in enumerate(codes)}

codes_dict
get_y_fn = lambda x: train_masks_path+f'{x.stem}_P{x.suffix}'
size = src_size//2



free = gpu_mem_get_free_no_cache()

# the max size of bs depends on the available GPU RAM

if free > 8200: 

    bs=8

else:           

    bs=4

print(f"using bs={bs}, have {free}MB of GPU RAM free")
src= (SegmentationItemList.from_folder(train_images_path).split_by_rand_pct()

      .label_from_func(get_y_fn, classes=codes))
data= (src.transform(get_transforms(), size=size, tfm_y=True)

      .databunch(bs=bs)

      )
data.show_batch(2)
data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)
name2id = {v:k for k,v in enumerate(codes)}

void_codes = [0,1,2,3,4,5,6]



#accuracy, not including void pixels

def acc_camvid(input, target):

    target = target.squeeze(1)

    mask = target not in void_codes

    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
metrics=acc_camvid

wd=1e-2



learn = unet_learner(data, models.resnet34, metrics=accuracy, wd=wd)
lr_find(learn)

learn.recorder.plot()