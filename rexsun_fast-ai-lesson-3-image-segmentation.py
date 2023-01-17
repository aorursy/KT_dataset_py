%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

import gc

import numpy as np

import pandas as pd

from fastai import *

from fastai.vision import * 

from fastai.metrics import error_rate,accuracy

from fastai.callbacks.hooks import *
# 我们想要的数据已经被放在了kaggle上

path = Path('/kaggle/input/repository/alexgkendall-SegNet-Tutorial-bb68b64/CamVid')
# 参看数据

path.ls()
fnames=get_image_files(path/"val")

fnames[:3]
lbl_names=get_image_files(path/"valannot")

lbl_names[:3]
# 我们查看一下第一张图片

img_f=fnames[0]

img=open_image(img_f)

img.show(figsize=(5,5))
def get_y_fn(x): return Path(str(x.parent)+"annot")/x.name



codes = array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',

    'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])
# 我们获得img_f的像素标记图片

mask=open_mask(get_y_fn(img_f))

mask.show(figsize=(5,5),alpha=1)
src_size=np.array(mask.shape[1:])

print(src_size)   # 显示数据的shape

mask.data  # 显示数据的内容

# 我们可以发现这个数据就在图片的每个像素上对原有图片的颜色进行标记
# 我们把所提取的图片的大小设置为原有像素的一半

bs,size=8,src_size//2
# data_block建立databunch

src=(SegmentationItemList.from_folder(path)

    .split_by_folder(valid="val")

    .label_from_func(get_y_fn,classes=codes))
data=(src.transform(get_transforms(),tfm_y=True)   # 对数据集进行增强

      .databunch(bs=bs)

      .normalize(imagenet_stats))
# 展示数据

data.show_batch(2,figsize=(10,7))
# 创建自己的评价指标

name2id={v:k for k,v in enumerate(codes)}

void_code=name2id["Void"]



def acc_camvid(input,target):

    target=target.squeeze()

    mask=target!=void_code

    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
metrics=acc_camvid

wd=1e-2
# 建模

learn=unet_learner(data,models.resnet34,metrics=metrics,wd=wd,model_dir="/kaggle/working/models")
# 获得学习率曲线

lr_find(learn)
# 查看学习率曲线

learn.recorder.plot()
# 设置学习率

lr=2e-4
learn.fit_one_cycle(10,slice(lr),pct_start=0.8)
learn.save("/kaggle/working/stage-1")
learn.show_results(3,figsize=(10,15))
# 我们对模型进行解冻，在进行一次训练

learn.load("/kaggle/working/stage-1")

learn.unfreeze()

learn.fit_one_cycle(10,max_lr=slice(lr//100,lr),pct_start=0.8)
learn.recorder.plot_losses()
learn.save("/kaggle/working/stage-2-unfreeze")
# 释放内存

del learn

gc.collect()
size=src_size

bs=8
data=(src.transform(get_transforms(),size=size,tfm_y=True)

     .databunch(bs=bs)

     .normalize(imagenet_stats))
# 建立模型，同时加载之前的预训练模型

learn=unet_learner(data,models.resnet34,metrics=metrics,wd=wd,model_dir="/kaggle/working/").load("stage-2-unfreeze")
# 寻找最佳的学习率

learn.lr_find(start_lr=1e-9)
learn.recorder.plot()
lr=2e-7

learn.fit_one_cycle(10,max_lr=slice(lr//10,lr),pct_start=0.8)
learn.save("/kaggle/working/stage-3-big")
learn.recorder.plot_losses()
learn.unfreeze()

learn.fit_one_cycle(1,max_lr=slice(lr//10,lr),pct_start=0.8)
learn.save("/kaggle/working/stage-4-big-unfreeze")
learn.show_results()
learn.summary()