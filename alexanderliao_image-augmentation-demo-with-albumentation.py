# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from PIL import Image
import cv2
# Any results you write to the current directory are saved as output.
!pip install albumentations
red = np.array(Image.open("../input/test/00631ec8-bad9-11e8-b2b9-ac1f6b6435d0_red.png").convert("L"))
green = np.array(Image.open("../input/test/00631ec8-bad9-11e8-b2b9-ac1f6b6435d0_green.png").convert("L"))
blue = np.array(Image.open("../input/test/00631ec8-bad9-11e8-b2b9-ac1f6b6435d0_blue.png").convert("L"))
yellow = np.array(Image.open("../input/test/00631ec8-bad9-11e8-b2b9-ac1f6b6435d0_yellow.png").convert("L"))
demo_rgb=Image.fromarray(np.concatenate((np.expand_dims(red,axis=2),np.expand_dims(green,axis=2),np.expand_dims(blue,axis=2)),axis=2))
demo_y=Image.fromarray(np.concatenate((np.expand_dims(yellow,axis=2),np.expand_dims(yellow,axis=2),np.expand_dims(blue,axis=2)),axis=2))
demo_y
demo_rgb
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate
)
def augment(aug, image):
    return aug(image=image)['image']

# Here's how to deal with 4-channel inputs

def augment_4chan(aug, image):
    image[:,:,0:3]=aug(image=image[:,:,0:3])['image']
    image[:,:,3]=aug(image=image[:,:,1:4])['image'][:,:,2]
    return image
aug = HorizontalFlip(p=1)
Image.fromarray(augment(aug,np.array(demo_rgb)))
aug = Blur(p=1,blur_limit=3)
Image.fromarray(augment(aug,np.array(demo_rgb)))
aug = ElasticTransform(p=1,border_mode=cv2.BORDER_REFLECT_101,alpha_affine=40)
Image.fromarray(augment(aug,np.array(demo_rgb)))
aug = RandomGamma(p=1)
Image.fromarray(augment(aug,np.array(demo_rgb)))
aug = RandomContrast(p=1)
Image.fromarray(augment(aug,np.array(demo_rgb)))
aug = RandomBrightness(p=1)
Image.fromarray(augment(aug,np.array(demo_rgb)))
aug = ChannelShuffle(p=1)
Image.fromarray(augment(aug,np.array(demo_rgb)))
aug = Rotate(p=1,limit=30)
Image.fromarray(augment(aug,np.array(demo_rgb)))
def strong_aug(p=1):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        #HueSaturationValue(p=0.3),
    ], p=p)
aug = strong_aug(p=1)
Image.fromarray(augment(aug,np.array(demo_rgb)))