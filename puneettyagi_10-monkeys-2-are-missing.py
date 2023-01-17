from glob import glob
PATH='../input/10-monkey-species/'
training_file_count=len(glob(f'{PATH}/training//training//**//*.*'))
val_file_count=len(glob(f'{PATH}/validation//validation//**//*.*'))
print(f'Images for Training : {training_file_count}')
print(f'Images for Validation : {val_file_count}')
! cat '{PATH}monkey_labels.txt'
from fastai.imports import *
from fastai.conv_learner import * 
from fastai.dataset import * 
from fastai.sgdr import *
from fastai.plots import * 
from fastai.transforms import *
from PIL import Image
import pandas as pd
train_files_path=glob(f'{PATH}/training//**//**/*.*')
val_files_path=glob(f'{PATH}/validation//**//**/*.*')
draw=np.random.choice(train_files_path,3)
for each in draw:
    im=Image.open(each)
    plt.figure()
    plt.imshow(np.asarray(im))
train_img_sz=pd.DataFrame([ Image.open(img).size for img in train_files_path],columns=['x','y'])
val_im_sz=pd.DataFrame([ Image.open(img).size for img in val_files_path],columns=['x','y'])
train_img_sz.mean(),train_img_sz.std()
val_im_sz.mean(),val_im_sz.mean()
train_img_sz.boxplot(figsize=(20,10))
val_im_sz.boxplot(figsize=(20,10))
sz=256
bs=64
arch=resnet34
def get_data(sz,bs):
    tfms=tfms_from_model(arch,sz,aug_tfms=transforms_side_on,max_zoom=1.2)
    return ImageClassifierData.from_paths('.',bs=bs,tfms=tfms,trn_name='../input/10-monkey-species/training/training',val_name='../input/10-monkey-species/validation/validation')
data=get_data(sz,bs)
import random
def draw_random_from_ds(data,n):
    x=random.sample(list(iter(data.trn_ds)),n)
    for each in x:
        a=each[0]
        for i,v in enumerate(a):
            a[i]=(a[i]-a[i].min())/(a[i].max()-a[i].min())
        a=a.reshape((a.shape[1],a.shape[2],3))
        plt.figure()
        plt.imshow(a)
    
draw_random_from_ds(data,3)
learn=ConvLearner.pretrained(arch,data,ps=.5,precompute=True)
learn.lr_find()
learn.sched.plot()
lr=6e-2
learn.fit(lr,3)
learn.fit(lr,3)
learn.precompute=False
learn.fit(lr,3)
learn.set_data(get_data(399,64))
learn.lr_find()
learn.sched.plot()
learn.fit(lr,3)
learn.set_data(get_data(512,64))
data=get_data(512,64)
draw_random_from_ds(data,3)
learn.fit(lr,3)
