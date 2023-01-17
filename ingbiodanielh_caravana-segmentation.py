import numpy as np 

import pandas as pd 



from fastai import *

from fastai.vision import *



import os

print(os.listdir("../input"))

path=Path('../input/train')
img_f =path.ls()[0]

img_f
img= open_image(img_f)

img
get_labels = lambda x: path.parent/f'train_masks/{x.stem}_mask.gif'

mask=open_mask(get_labels(img_f), div=True)

mask
plt.imshow(img.data.transpose(1,2).numpy().T)

plt.imshow(mask.data[0])
class CaravanaSegmentationLabelList(SegmentationLabelList):

    def open(self,fn): return open_mask(fn, div=True)

    

class CaravanaSegmentationItemList(ImageList):

    _label_cls= CaravanaSegmentationLabelList
src=(CaravanaSegmentationItemList.from_folder(path)

     .split_by_rand_pct()

     .label_from_func(get_labels, classes=['void','car']))
data = (src.transform(get_transforms(), size=(224,224), tfm_y=True)

        .databunch(bs=16)

        .normalize(imagenet_stats))
data.show_batch(rows=3, alpha=0.6)
im,m=data.one_batch()

im.shape, m.shape
m[0,0].unique()
def dice(input:Tensor, targs:Tensor, iou:bool=False)->Rank0Tensor:

    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."

    n = targs.shape[0]

    input = input.argmax(dim=1).view(n,-1)

    targs = targs.view(n,-1)

    intersect = (input*targs).sum().float()

    union = (input+targs).sum().float()

    if not iou: return 2. * intersect / union

    else: return intersect / (union-intersect+1.0)

    

# def iou(outputs: torch.Tensor, labels: torch.Tensor):

#     # You can comment out this line if you are passing tensors of equal shape

#     # But if you are passing output from UNet or something it will most probably

#     # be with the BATCH x 1 x H x W shape

# #     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    

#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0

#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    

#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    

#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    

#     return thresholded

def accuracy_carvana(input, target):

    target=target.squeeze(1)

    mask =target>0

    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
learner= unet_learner(data, models.resnet34, metrics=[dice, accuracy_carvana], model_dir="/tmp/models/") 
# pred, target = learner.model(im.cuda()), m.cuda()

# pred.shape, target.shape
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(1, 1e-3)
learner.save('stage-1')
learner.load('stage-1');
img,mk= learner.data.one_batch()
def overlay(pred, mask):

    plt.imshow(pred[0].transpose(1,2).numpy().T, cmap='gray')

    plt.imshow(mask[0,0].numpy(),cmap='jet', alpha=0.5)
overlay(img,mk)