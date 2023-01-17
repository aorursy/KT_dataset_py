# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

import seaborn as sns

import matplotlib.pyplot as plt

import os

import cv2

import glob

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pathlib import Path

from fastai import *

from fastai.vision import *

import torch

from fastai.callbacks.hooks import *

from tqdm import tqdm_notebook
data_folder = Path("../input/hackerearth-dl-challengeautotag-images-of-gala/dataset")

data_path = "../input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images"

path = os.path.join(data_path , "*jpg")

path
files = glob.glob(path)

data=[]

name_img_mapper = {}

for file in tqdm_notebook(files):

    fn = file.split('/')[-1]

    image = cv2.imread(file)

    data.append(file)

    name_img_mapper[fn] = image
## read the csv data files

train_df = pd.read_csv('../input/hackerearth-dl-challengeautotag-images-of-gala/dataset/train.csv')

test_df = pd.read_csv('../input/hackerearth-dl-challengeautotag-images-of-gala/dataset/test.csv')
train_df.groupby('Class').count()
sns.countplot(x='Class' , data=train_df)
train_images = data
category = {'Food': 1, 

'misc': 2, 

'Attire': 3, 

'Decorationandsignage': 4}



train_df.loc[train_df['Class']== 'Food']['Image'][:3].tolist()
train_df.loc[train_df['Class']== 'Food']['Image'][:3]
def plot_class(cat):

    

    fetch = train_df.loc[train_df['Class']== cat][:3]

    images_names = train_df.loc[train_df['Class']== cat]['Image'][:3].tolist()

#     print(images_names)

    fig = plt.figure(figsize=(20,15))

    

    for i, img_name in enumerate(images_names):

#         print(i, img_name)

        plt.subplot(1,3 ,i+1)

        plt.imshow(name_img_mapper[img_name])

#         plt.xlabel(cat + " (Index:" +str()+")" )

    plt.show()
plt.imshow(name_img_mapper['image7042.jpg'])
plot_class('Decorationandsignage')
##transformations to be done to images

tfms = get_transforms(do_flip=True,flip_vert=False ,max_rotate=15.0, max_zoom=1.2, max_lighting=0.5, max_warp=0.1, p_affine=0.2,

                      p_lighting=0.55)

#, xtra_tfms=zoom_crop(scale=(0.9,1.8), do_rand=True, p=0.8))



## create databunch of test set to be passed

test_img = ImageList.from_df(test_df, path=data_folder, folder='Test Images')
np.random.seed(145)

## create source of train image databunch

src = (ImageList.from_df(train_df, path=data_folder, folder='Train Images')

       .split_by_rand_pct(0.15)

       #.split_none()

       .label_from_df()

       .add_test(test_img))
data = (src.transform(tfms, size=299,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)

        .databunch(path='.', bs=32, device= torch.device('cuda:0')).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,12))
print(data.classes)
learn = cnn_learner(data=data, base_arch=models.densenet201, metrics=[FBeta(beta=1, average='macro'), accuracy],

                    callback_fns=ShowGraph)
#lets find the correct learning rate to be used from lr finder

learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-03

#learn.fit_one_cycle(10, slice(lr))

learn.fit_one_cycle(8, slice(lr), wd=0.25)
# lr = 1.2e-03

# #learn.fit_one_cycle(10, slice(lr))

# learn.fit_one_cycle(8, slice(lr), wd=0.25)
#lets plot the lr finder record

learn.unfreeze()

learn.lr_find()



learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10,slice(1e-06,lr/8),wd=0.25)
#lets see the most mis-classified images (on validation set)

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,10))
# learn.recorder.plot_losses()
##learn.TTA improves score further. lets see for the validation set

pred_val,y = learn.TTA(ds_type=DatasetType.Valid)

from sklearn.metrics import f1_score, accuracy_score

valid_preds = [np.argmax(pred_val[i])+1 for i in range(len(pred_val))]

valid_preds = np.array(valid_preds)

y = np.array(y+1)

accuracy_score(valid_preds,y),f1_score(valid_preds,y, average='weighted')
preds,_ = learn.TTA(ds_type=DatasetType.Test)

#preds,_ = learn.get_preds(ds_type = DatasetType.Test)

labelled_preds = [np.argmax(preds[i])+1 for i in range(len(preds))]



labelled_preds = np.array(labelled_preds)
df = pd.DataFrame({'Image':test_df['Image'], 'Class':labelled_preds})

df
category = {'Food': 3, 

'misc': 4, 

'Attire': 1, 

'Decorationandsignage': 2}

rev_category = {val: key for key, val in category.items()}

df['Class'] = df['Class'].map(rev_category)

df.to_csv('submission.csv', index=False)
df.Class.value_counts()
from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



create_download_link(filename = 'submission.csv')