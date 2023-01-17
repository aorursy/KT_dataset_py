!pip install "torch==1.4" "torchvision==0.5.0"
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.metrics import error_rate
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from shutil import copyfile
copyfile(src = "../input/cutmix/shared/0_image_data_augmentation/exp/nb_new_data_augmentation.py" , 
         dst = "../working/nb_new_data_augmentation.py")
from nb_new_data_augmentation import *
path = Path('../input/shopee-round-2-product-detection-challenge')
path
# change filename and category name
# i chose 1000 data for each group
N = 1000 #number of data in each category
df = pd.read_csv('/content/train.csv')
df = df.groupby('category').apply(lambda x: x[:N]).reset_index(drop=True)
df['filename'] = 'train/' + df['category'].apply(lambda x: '{0:0>2}'.format(x)).apply(str)+'/'+df['filename']

df['category'] = df['category'].replace({0: '00 Dress',1: '01 Sarung',2: '02 Shirt(Top)',3: '03 Long Sleeves/Hoodie',4: '04 Jeans(Female)',5: '05 Ring',6: '06 Ear Rings',7: '07 Cap',8: '08 Purse/Wallet',9: '09 Bags',10: '10 Phone Cover',11: '11 Phone',12: '12 Clock',13: '13 Plastic Baby Bottles',14: '14 Rice Cooker',15: '15 Coffee',16: '16 Shoe',17: '17 High Heels',18: '18 Aircon',19: '19 Pendrive',20: '20 Chair',21: '21 Racket',22: '22 Helmet',23: '23 Gloves',24: '24 Watch',25: '25 Belt',26: '26 Headphones/Earpiece',27: '27 Toy Car',28: '28 Suit/Jacket (Male)',29: '29 Tuxedo Pants (Male)',30: '30 Sport Shoes',31: '31 Biscuit/ Junk Food',32: '32 Face Mask',33: '33 Sanitizer/Antiseptic',34: '34 Skin Care(?)',35: '35 Perfume/Cologne',36: '36 Cleaning Supplies',37: '37 Laptop',38: '38 Bowls',39: '39 Vases?',40: '40 Shower Stuff(Showerhead)',41: '41 Sofa',})
df
# batchsize
bs = 64
# Loading data using ImageDataBunch.from_df
np.random.seed(0)
data = ImageDataBunch.from_df(path, df, folder='/train/train',valid_pct=0.2,
                              test='test/test/test',
                              ds_tfms=get_transforms(), 
                              size=224, bs=bs
                              ).normalize(imagenet_stats)
# checking data
data.show_batch(rows=3, figsize=(10, 5))
# using model resnet50 and applying cutmix in the image
learn = cnn_learner(data, models.resnet50, 
                    metrics=[error_rate, accuracy]
                   ).cutmix().show_multi_img_tfms()
learn.model_dir = '/kaggle/working/'
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(8, max_lr=2e-3)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(8, max_lr=slice(1e-5 , lr/5))
# Predict Using TTA
preds, y = learn.TTA(ds_type=DatasetType.Test)
category = [data.classes[int(x)][:2] for x in np.argmax(preds, 1)]
filename = []
num = len(learn.data.test_ds)
for i in range(num):
    filename.append(str(learn.data.test_ds.items[i]).split('/')[-1])
test_csv = pd.read_csv('../input/shopee-round-2-product-detection-challenge/test.csv')
submit = pd.DataFrame({'filename':filename,'category':category})
submit = submit[submit['filename'].isin(test_csv['filename'])]
submit
submit.to_csv('submission.csv', header=True, index=False)