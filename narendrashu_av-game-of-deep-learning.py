# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# from sklearn.metrics import f1_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

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
## set the data folder

data_folder = Path("../input")
data_path = "../input/train/images/"

path = os.path.join(data_path , "*jpg")
files = glob.glob(path)

data=[]

for file in files:

    image = cv2.imread(file)

    data.append(image)
## read the csv data files

train_df = pd.read_csv('../input/train/train.csv')

test_df = pd.read_csv('../input/test_ApKoW4T.csv')

submit = pd.read_csv('../input/sample_submission_ns2btKE.csv')
train_df.shape, test_df.shape
train_df.head()
test_df.head()
# submit.head()
train_df.groupby('category').count()
sns.countplot(x='category' , data=train_df)
train_images = data[:6252]

test_images= data[6252:]
## mapping the ship categories  

category = {'Cargo': 1, 

'Military': 2, 

'Carrier': 3, 

'Cruise': 4, 

'Tankers': 5}
def plot_class(cat):

    

    fetch = train_df.loc[train_df['category']== category[cat]][:3]

    fig = plt.figure(figsize=(20,15))

    

    for i , index in enumerate(fetch.index ,1):

        plt.subplot(1,3 ,i)

        plt.imshow(train_images[index])

        plt.xlabel(cat + " (Index:" +str(index)+")" )

    plt.show()
plot_class('Cargo')
plot_class('Military')
plot_class('Carrier')
plot_class('Tankers')
plot_class('Cruise')
# doc(src.transform)
##transformations to be done to images

tfms = get_transforms(do_flip=False,flip_vert=False ,max_rotate=10.0, max_zoom=1.22, max_lighting=0.22, max_warp=0.0, p_affine=0.75,

                      p_lighting=0.75)

#, xtra_tfms=zoom_crop(scale=(0.9,1.8), do_rand=True, p=0.8))



## create databunch of test set to be passed

test_img = ImageList.from_df(test_df, path=data_folder/'train', folder='images')
np.random.seed(145)

## create source of train image databunch

src = (ImageList.from_df(train_df, path=data_folder/'train', folder='images')

       .split_by_rand_pct(0.2)

       #.split_none()

       .label_from_df()

       .add_test(test_img))
data = (src.transform(tfms, size=299,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)

        .databunch(path='.', bs=32, device= torch.device('cuda:0')).normalize(imagenet_stats))



# data = (src.transform(tfms, size=484,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)

#         .databunch(path='.', bs=16, device= torch.device('cuda:0')).normalize(imagenet_stats))
## lets see the few images from our databunch

data.show_batch(rows=3, figsize=(12,12))
print(data.classes)
# doc(cnn_learner)
#lets create learner. tried with resnet152, densenet201, resnet101

learn = cnn_learner(data=data, base_arch=models.resnet101, metrics=[FBeta(beta=1, average='macro'), accuracy],

                    callback_fns=ShowGraph)



# learn = cnn_learner(data=data, base_arch=models.densenet161, metrics=[FBeta(beta=1, average='macro'), accuracy],

#                     callback_fns=ShowGraph).mixup()
#learn.opt_func = optim.Adam

#learn.crit = FocalLoss()

# learn_gen = None

# gc.collect()

# torch.cuda.empty_cache()

learn.summary()
#lets find the correct learning rate to be used from lr finder

learn.lr_find()

learn.recorder.plot(suggestion=True)
#lets start with steepset slope point. adding wd (weight decay) not to overfit as we are running 15 epochs 

lr = 3e-03

#learn.fit_one_cycle(10, slice(lr))

learn.fit_one_cycle(15, slice(lr), wd=0.2)
#lets plot the lr finder record

learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
# train for  more cycles after unfreezing

learn.fit_one_cycle(10,slice(1e-05,lr/8),wd=0.15)

#learn.fit_one_cycle(10, slice(5e-06, lr/8))
learn.freeze_to(-3)
## finding the LR

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(6, slice(1e-06, lr/10),wd=0.1)
## freezing initial all layers except last 2 layers

learn.freeze_to(-2)
## training for few cylcles more

learn.fit_one_cycle(6, slice(5e-07, lr/20),wd=0.1)
learn.freeze_to(-1)
## training even more

learn.fit_one_cycle(5, slice(1e-07, lr/30),wd=0.05)
learn.fit_one_cycle(6, slice(1e-07, lr/100))
#lets see the most mis-classified images (on validation set)

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(7,6))
learn.recorder.plot_losses()
interp.plot_confusion_matrix(figsize=(6,6), dpi=60) ## on validation set
interp.most_confused(min_val=4) ## on validation set
idx=1

x,y = data.valid_ds[idx]

x.show()
k = tensor([

    [0.  ,-5/3,1],

    [-5/3,-5/3,1],

    [1.  ,1   ,1],

]).expand(1,3,3,3)/6
t = data.valid_ds[1][0].data; t.shape
edge = F.conv2d(t[None], k)
show_image(edge[0], figsize=(5,5));
m = learn.model.eval();

xb,_ = data.one_item(x)

xb_im = Image(data.denorm(xb)[0])

xb = xb.cuda()
def hooked_backward(cat=y):

    with hook_output(m[0]) as hook_a: 

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(xb)

            preds[0,int(cat)].backward()

    return hook_a,hook_g
hook_a,hook_g = hooked_backward()

acts  = hook_a.stored[0].cpu()

acts.shape
avg_acts = acts.mean(0)

avg_acts.shape

torch.Size([11, 11])
def show_heatmap(hm):

    _,ax = plt.subplots()

    xb_im.show(ax)

    ax.imshow(hm, alpha=0.6, extent=(0,352,352,0),

              interpolation='bilinear', cmap='magma');
show_heatmap(avg_acts)
##learn.TTA improves score further. lets see for the validation set

pred_val,y = learn.TTA(ds_type=DatasetType.Valid)

from sklearn.metrics import f1_score, accuracy_score

valid_preds = [np.argmax(pred_val[i])+1 for i in range(len(pred_val))]

valid_preds = np.array(valid_preds)

y = np.array(y+1)

accuracy_score(valid_preds,y),f1_score(valid_preds,y, average='micro')
preds,_ = learn.TTA(ds_type=DatasetType.Test)

#preds,_ = learn.get_preds(ds_type = DatasetType.Test)

labelled_preds = [np.argmax(preds[i])+1 for i in range(len(preds))]



labelled_preds = np.array(labelled_preds)
#create submission file

df = pd.DataFrame({'image':test_df['image'], 'category':labelled_preds}, columns=['image', 'category'])

df.to_csv('submission.csv', index=False)
## function to create download link

from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
create_download_link(filename = 'submission.csv')
df.category.unique()
df.head()