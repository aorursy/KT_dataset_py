import numpy as np

import pandas as pd
!pip install "torch==1.4" "torchvision==0.5.0"

import torch

torch.__version__
from fastai.vision import *

from fastai.metrics import error_rate
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from shutil import copyfile

copyfile(src = "../input/cutmix/shared/0_image_data_augmentation/exp/nb_new_data_augmentation.py" , 

         dst = "../working/nb_new_data_augmentation.py")
from nb_new_data_augmentation import *
path = Path('../input/shopee-product-detection-student')
# change filename and category name

# chose 1000 data for each category



N = 1000 #number of data in each category

df = pd.read_csv('../input/shopee-product-detection-student/train.csv')



df = df.groupby('category').apply(lambda x: x[:N]).reset_index(drop=True) # choose first till N(th) image data from df

df['filename'] = 'train/' + df['category'].apply(lambda x: '{0:0>2}'.format(x)).apply(str)+'/'+df['filename'] # change file name column to the filepath
# change label name

df['category'] = df['category'].replace({0: '00 Dress',1: '01 Sarung',2: '02 Shirt(Top)',3: '03 Long Sleeves/Hoodie',4: '04 Jeans(Female)',5: '05 Ring',6: '06 Ear Rings',7: '07 Cap',8: '08 Purse/Wallet',9: '09 Bags',10: '10 Phone Cover',11: '11 Phone',12: '12 Clock',13: '13 Plastic Baby Bottles',14: '14 Rice Cooker',15: '15 Coffee',16: '16 Shoe',17: '17 High Heels',18: '18 Aircon',19: '19 Pendrive',20: '20 Chair',21: '21 Racket',22: '22 Helmet',23: '23 Gloves',24: '24 Watch',25: '25 Belt',26: '26 Headphones/Earpiece',27: '27 Toy Car',28: '28 Suit/Jacket (Male)',29: '29 Tuxedo Pants (Male)',30: '30 Sport Shoes',31: '31 Biscuit/ Junk Food',32: '32 Face Mask',33: '33 Sanitizer/Antiseptic',34: '34 Skin Care(?)',35: '35 Perfume/Cologne',36: '36 Cleaning Supplies',37: '37 Laptop',38: '38 Bowls',39: '39 Vases?',40: '40 Shower Stuff(Showerhead)',41: '41 Sofa',})

df
# Loading data using ImageDataBunch.from_df

np.random.seed(0)

data = ImageDataBunch.from_df(path, df, folder='/train/train',valid_pct=0.2, # add train data and split 20% to valid

                              test='test/test/test', #add train data

                              ds_tfms=get_transforms(), # apply transorm by default

                              size=224, bs=64 # change size to 224 and batch size to 64

                              ).normalize(imagenet_stats) # normalize data
# checking data

data.show_batch(rows=3, figsize=(10, 5))
# using model resnet50 and applying cutmix in the image

learn = cnn_learner(data, models.resnet50, 

                    metrics=[error_rate, accuracy]

                   ).cutmix().show_multi_img_tfms()
learn.model_dir = '/kaggle/working/'
# to find a good learning rate

learn.lr_find()

learn.recorder.plot(suggestion=True)
# fit one cycle, set max learning rate

learn.fit_one_cycle(8, max_lr=slice(2e-3))
# unfreeze

learn.unfreeze() 
learn.lr_find()

learn.recorder.plot(suggestion=True)
# fit one cycle again, set max learning rate

learn.fit_one_cycle(8, max_lr=slice(1e-5 , 4e-4))
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)
# the most confuse data

interp.most_confused(min_val=40)
# the most confuse matrix

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
# Predict Using TTA

preds, y = learn.TTA(ds_type=DatasetType.Test)
# make category column for submission

category = [data.classes[int(x)][:2] for x in np.argmax(preds, 1)]
# make filename column for submission

filename = []

num = len(learn.data.test_ds)

for i in range(num):

    filename.append(str(learn.data.test_ds.items[i]).split('/')[-1])
# make submission's dataframe

submit = pd.DataFrame({'filename':filename,'category':category}) 

#the test folder have extra image data, so drop it

test_csv = pd.read_csv('../input/shopee-product-detection-student/test.csv')

submit = submit[submit['filename'].isin(test_csv['filename'])]

submit
submit.to_csv('submission.csv', header=True, index=False)