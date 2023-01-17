# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os, sys

print(os.listdir("../input/aptos2019-blindness-detection/"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from zipfile import ZipFile

from fastai.vision import *

from fastai.metrics import error_rate

from fastai.callbacks import *

import cv2
bs = 64



!ls
# copy pretrained weights for resnet50 to the folder fastai will search by default

Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'


#doc(ImageDataBunch)
PATH = Path('../input/aptos2019-blindness-detection')



df_train = pd.read_csv(PATH/'train.csv', dtype={'id_code':str, 'diagnosis':int})

df_test = pd.read_csv(PATH/'test.csv')

df_train['zc'] = 0

df_train['zc'].loc[(df_train['diagnosis']==0)] = 0

df_train['zc'].loc[~(df_train['diagnosis']==0)]= 1

df_train['oc'] = 0

df_train['oc'].loc[(df_train['diagnosis']==1)] = 0

df_train['oc'].loc[~(df_train['diagnosis']==1)]= 1

df_train['tc'] = 0

df_train['tc'].loc[(df_train['diagnosis']==2)] = 0

df_train['tc'].loc[~(df_train['diagnosis']==2)]= 1

df_train['thc'] = 0

df_train['thc'].loc[(df_train['diagnosis']==3)] = 0

df_train['thc'].loc[~(df_train['diagnosis']==3)]= 1
print(df_train.head(5))

df_train.diagnosis.value_counts()

df_train.hist()
# # create Stratified validation split (12.50%)

# #fastai does not include stratify option in train test data split, however according to the lecturer, 

# #imbalance classifiers will be handle by the deep learning quite well, not sure this is true in this case

# from sklearn.model_selection import StratifiedKFold

# cv = StratifiedKFold(n_splits=8, random_state=42)

# tr_ids, val_ids = next(cv.split(df_train.id_code, df_train.diagnosis))

# print(len(tr_ids), len(val_ids))

# _ = df_train.loc[val_ids].hist()
# print(val_ids)

# print(tr_ids)
# import zipfile

# with zipfile.ZipFile('./train_images.zip', 'r') as zip_ref:

#     zip_ref.extractall('./train_images')
# with zipfile.ZipFile('./test_images.zip', 'r') as zip_ref:

#     zip_ref.extractall('./test_images')
# def crop_image_from_gray(img,tol=7):

#     if img.ndim ==2:

#         mask = img>tol

#         return img[np.ix_(mask.any(1),mask.any(0))]

#     elif img.ndim==3:

#         gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#         mask = gray_img>tol

        

#         check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

#         if (check_shape == 0): # image is too dark so that we crop out everything,

#             return img # return original image

#         else:

#             img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

#             img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

#             img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

#     #         print(img1.shape,img2.shape,img3.shape)

#             img = np.stack([img1,img2,img3],axis=-1)

#     #         print(img.shape)

#         return img

    

# IMG_SIZE = 512



# def _load_format(path, convert_mode, after_open)->Image:

#     image = cv2.imread(path)

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     image = crop_image_from_gray(image)

#     image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

#     image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0), 10) ,-4 ,128)

                    

#     return Image(pil2tensor(image, np.float32).div_(255)) #return fastai Image format



# vision.data.open_image = _load_format
# create image data bunch

# create first image list using datafram, than split train and valid dataset according to stratified indx and lst lable imagelist with classes

data = ImageDataBunch.from_df('./', 

                              df=df_train, 

                              valid_pct=0.2,

#                               folder="../input/aptos2015/resizedtrain15",

                              folder="../input/aptos2019-blindness-detection/train_images",

                              suffix=".png",

                              ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                              size=224,

                              bs=128, 

                              num_workers=32,

                             label_col='zc').normalize(imagenet_stats)
#create data using fastai ImageDataBunch function, create from image list with lable.

#simple data augmentation with flip and rotate since this is an eyeball image, the image is normalized using default imagenet_stats, another possible 

#option would be use the aptos19_stats, which not sure how to derive from yet

# data.show_batch(rows=3, figsize=(7,6))
data.classes
kappa = KappaScore()

kappa.weights = "quadratic"

learnz = cnn_learner(data, models.resnet50, metrics=[error_rate, kappa],

                    callback_fns = [

                                partial(EarlyStoppingCallback, monitor='kappa_score', min_delta=0.001, patience=2),

                                partial(ReduceLROnPlateauCallback),

#                               partial(GradientClipping, clip=0.2),

                                partial(SaveModelCallback, every = 'improvement', monitor='kappa_score', name='bestmodel')],

                    model_dir="/tmp/model/")
learnz.lr_find()

learnz.recorder.plot(suggestion=True)
learnz.fit_one_cycle(10,1e-2)
learnz.unfreeze()

learnz.fit_one_cycle(15,slice(1.32e-6,1.32e-3))
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learnz.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))
preds,y = learnz.get_preds(ds_type=DatasetType.Test)

sample_df.diagnosis = preds.argmax(1)

sample_df.diagnosis.value_counts()
# log_preds,y = learn2.TTA(ds_type=DatasetType.Test)
# sample_df.diagnosis = np.argmax(log_preds.numpy(), axis=1)

# sample_df.head(50)

# sample_df.diagnosis.hist()

# sample_df.diagnosis.value_counts()
# sample_df.to_csv('submission.csv',index=False)
!mv {learnz.model_dir}/*.pth .

os.listdir()