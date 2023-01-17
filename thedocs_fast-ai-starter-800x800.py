# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

import matplotlib.pyplot as plt

from fastai.metrics import accuracy, KappaScore

from fastai.vision import *

from fastai import *

from fastai.callbacks import *

from skimage import io

from fastai.vision.image import *

import cv2

from PIL import Image

from imblearn import over_sampling

import shutil







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
data_dir = '../input/aptos2019-blindness-detection'

train_df = pd.read_csv(os.path.join(data_dir,'train.csv'))

print('Train df: ')

print(train_df.head(4))



add_extension = lambda x: str(x) + '.png'

add_dir = lambda x: os.path.join('train_images', x)



train_df['id_code'] = train_df['id_code'].apply(add_extension)

train_df['id_code'] = train_df['id_code'].apply(add_dir)



fname = train_df['id_code'].iloc[:4]



data_dir = Path(data_dir)

train_dir = data_dir/'train_images'

im1 = io.imread(str(data_dir/fname[0]))

im2 = io.imread(str(data_dir/fname[1]))

im3 = io.imread(str(data_dir/fname[2]))

im4 = io.imread(str(data_dir/fname[3]))



plt.subplot(2,2,1)

plt.imshow(im1)

plt.subplot(2,2,2)

plt.imshow(im2)

plt.subplot(2,2,3)

plt.imshow(im3)

plt.subplot(2,2,4)

plt.imshow(im4)

plt.show()



print(str(len(train_df)) + ' number of samples')

val_counts = train_df['diagnosis'].value_counts()

print('Distribution of classes',val_counts)



print('modified train df: ')

train_df.head(2)
#def _preprocess(im):

#    "Flip `x` horizontally."

    

#    im = cv2.addWeighted (np.array(x),4, cv2.GaussianBlur(np.array(x) , (0,0) , 800/10) ,-4 ,128)

#    #im = np.array(im) * 100

#    return im



#preprocess = TfmPixel(_preprocess)
dict_rs={0: 1805, 2: 1200, 1: 600, 3: 500, 4: 550}

ros = over_sampling.RandomOverSampler(dict_rs,random_state=42)



X_res, y_res = ros.fit_resample(train_df['id_code'].values.reshape(-1,1), train_df['diagnosis'].values)



df_new = pd.DataFrame(columns=['id_code', 'diagnosis'])

df_new['diagnosis'] = y_res

df_new['id_code'] = X_res

train_df = df_new.copy()

train_df.diagnosis.value_counts() 

# data augmentation 

tfms = get_transforms(max_rotate=80, flip_vert=True)

bs = 8

PATH = Path('../input/aptos2019-blindness-detection')



data = (

    ImageList.from_df(train_df,PATH)

        .split_by_rand_pct(0.1, seed=42)

        .label_from_df()

        .transform(tfms,size=800)

        .databunch(bs=8)

        .normalize(imagenet_stats)

    

    )

data.classes 
cwd = os.getcwd()

# creating learner with resent architecture 



kappa = KappaScore()

kappa.weights = "quadratic"



Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'



learn = cnn_learner(data, models.resnet34, metrics=[accuracy, kappa], pretrained=True, 

                    callback_fns=[partial(CSVLogger, append=True)], path='../tmp/model/')



learn.fit_one_cycle(5, slice(0.01))



learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(7, slice(1e-6, 0.01/6))



learn.save('model1')



shutil.copy('../tmp/model/model1.pth', os.getcwd())

#learn.fit_one_cycle(5, max_lr = 1.5e-6)



#learn.freeze()

#learn.fit_one_cycle(2, max_lr = 1.5e-6)
sample_df = pd.read_csv(PATH/'sample_submission.csv')

learn.data.add_test(ImageList.from_df(sample_df,PATH,folder='test_images',suffix='.png'))



preds,y = learn.get_preds(DatasetType.Test)



sample_df.diagnosis = preds.argmax(1)

sample_df.head()



sample_df.to_csv('submission.csv',index=False)