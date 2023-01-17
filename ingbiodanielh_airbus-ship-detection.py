



import numpy as np 

import pandas as pd 



from fastai import *

from fastai.vision import *

from fastai.callbacks import *



import os

print(os.listdir("../input"))



df= pd.read_csv('../input/train_ship_segmentations_v2.csv').dropna().set_index('ImageId')

df.head()
path = Path('../input/')

(path).ls()
img_f= path/'train_v2/000155de5.jpg'

# df.loc['000155de5.jpg']

open_image(img_f)
mask=open_mask_rle(df.loc['000155de5.jpg'].values[0], shape=(768,768))

plt.imshow(mask.data.transpose(1,2)[0])
class ShipSegmentationLabelList(SegmentationLabelList):

    def open(self,fn): 

        def open_mask_rle_T(mask_rle:str, shape:Tuple[int, int])->ImageSegment:

            "Return `ImageSegment` object create from run-length encoded string in `mask_lre` with size in `shape`."

            x = FloatTensor(rle_decode(str(mask_rle), shape).astype(np.uint8).T)

            x = x.view(shape[1], shape[0], -1)

            return ImageSegment(x.permute(2,0,1))

        return open_mask_rle_T(fn, shape=(768, 768))

    

class ShipSegmentationItemList(ImageList):

    _label_cls= ShipSegmentationLabelList
get_labels= lambda x: df.loc[x.parts[-1]].values[0][0]
train_files= [Path(os.path.join(path/'train_v2',f))  for f in df.index]
src=(ShipSegmentationItemList(train_files)

        .split_by_rand_pct()

        .label_from_func(get_labels,classes=['water','ship']))
data = (src.transform(get_transforms(flip_vert=True), size=(224,224), tfm_y=True)

        .databunch(bs=16, num_workers=2)

        .normalize(imagenet_stats))
data.show_batch(rows=2, alpha=0.7, figsize=(15,15)) 
im,m = data.one_batch()

im.shape,m.shape

plt.imshow(im[0].transpose(1,2).numpy().T, cmap='gray')

plt.imshow(m[0][0], cmap='ocean', alpha=0.5)
def dice(input:Tensor, targs:Tensor, iou:bool=False)->Rank0Tensor:

    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."

    n = targs.shape[0]

    input = input.argmax(dim=1).view(n,-1)

    targs = targs.view(n,-1)

    intersect = (input*targs).sum().float()

    union = (input+targs).sum().float()

    if not iou: return 2. * intersect / union

    else: return intersect / (union-intersect+1.0)



def accuracy_ship(input, target):

    target=target.squeeze(1)

    mask =target>0

    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
learner= unet_learner(data, models.resnet34,metrics=[accuracy_ship,dice], 

                      model_dir="/tmp/models/", 

                      callback_fns=[partial(SaveModelCallback,every='epoch',name='1'),

                                 ShowGraph])
learner.lr_find()

learner.recorder.plot()
lr=1e-3

learner.fit_one_cycle(4, slice(1e-4,2*lr), wd=1e-2)

MODEL = '/kaggle/working/model'