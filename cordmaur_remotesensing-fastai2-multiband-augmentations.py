# update torch and torch vision

!pip install -q torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# install kornia, we will give it a try to accelarate our preprocessing

!pip install -q --upgrade kornia

!pip install -q allennlp==1.1.0.rc4
# and install fastai2

!pip install -q --upgrade fastai
import torch

print(torch.__version__)

print(torch.cuda.is_available())



import fastai

print(fastai.__version__)



# other imports

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from PIL import Image



from fastai.vision.all import *
import torch

import fastai

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from PIL import Image



from fastai.vision.all import *



def open_tif(fn, chnls=None, cls=torch.Tensor):

    im = (np.array(Image.open(fn))).astype('float32')

    return cls(im)



# The map_filename function makes it easier to map from one folder to another by replacing strings

def map_filename(base_fn, str1, str2):

    return Path(str(base_fn).replace(str1, str2))



def get_filenames(red_filename):

    return [red_filename,

            map_filename(red_filename, str1='red', str2='green'),

            map_filename(red_filename, str1='red', str2='blue'),

            map_filename(red_filename, str1='red', str2='nir'),

           ]





# the open multi-spectral tif function will be in charge of opening the separate tifs and collate them

def open_ms_tif(files):

    ms_img = None

    

    for path in files:

        img = open_tif(path)

        

        if ms_img is None:

            ms_img = img[None]

        else:

            ms_img = np.concatenate([ms_img, img[None]], axis=0)

            

    return TensorImage(ms_img)

    

# get items from both datasets

items_95 = get_files('/kaggle/input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/train_red_additional_to38cloud', extensions='.TIF')

items_38 = get_files('/kaggle/input/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training/train_red/', extensions='.TIF')

all_items = items_95 + items_38
# now select just the non empty ones

n_empty = pd.read_csv('/kaggle/input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/training_patches_95-cloud_nonempty.csv')



def non_empty(item):

    

    if n_empty.name.isin([item.stem[4:]]).any():

        return True

    else:

        return False

    

items_mask = all_items.map(non_empty)

items = all_items[items_mask]

items
idx=5

img_pipe = Pipeline([get_filenames, open_ms_tif])

img = img_pipe(items[idx])



mask_pipe = Pipeline([partial(map_filename, str1='red', str2='gt'), 

                      partial(open_tif, cls=TensorMask)])



mask = mask_pipe(items[idx])

print(img.shape, mask.shape)



_, ax = plt.subplots(1, 2, figsize=(12,5))

ax[0].imshow(img.permute(1, 2, 0)[..., :3]/20000)

mask.show(ctx=ax[1])
def show_img(tensor_img, ctx=None):

    ctx = plt.subplot() if ctx is None else ctx

    

    #normalize to fit between 0 and 1

    if tensor_img.max() > 0:

        tensor_img = tensor_img / tensor_img.max()

    

    ctx.imshow(tensor_img.permute(1, 2, 0)[..., :3])

    

# To create this DataBlock we don't need to specify the get_items function 

# because we will pass the list of files as the source

db = DataBlock(blocks=(TransformBlock([get_filenames, open_ms_tif, lambda x: x/10000]), 

                       TransformBlock([partial(map_filename, str1='red', str2='gt'), 

                                       partial(open_tif, cls=TensorMask)])),

               splitter=RandomSplitter(valid_pct=0.2, seed=0)

              )



# Now We could call db.summary() to see if everything goes well

# %time db.summary(source=items)

# Instead, we will create the dataloader and display a batch sample

    

ds = db.datasets(source=items)

dl = db.dataloaders(source=items, bs=4)

batch = dl.one_batch()

print(batch[0].shape, batch[1].shape)



# # display the batch

# _, ax = plt.subplots(2, batch[0].shape[0], figsize=(batch[0].shape[0]*4, 7))

# for i in range(batch[0].shape[0]):

    

#     show_img(batch[0][i], ctx=ax[0, i])

#     TensorMask(batch[1][i]).show(ax[1, i])
import albumentations as A
import pdb

class SegmentationAlbumentationsTransform(ItemTransform):

#     split_idx=0

    def __init__(self, aug, **kwargs): 

        super().__init__(**kwargs)

        self.aug = aug

        

    def encodes(self, x):

        img,mask = x

        

        img = img/img.max()

        

#         print('applying augmentation')

        # for albumentations to work correctly, the channels must be at the last dimension

        aug = self.aug(image=np.array(img.permute(1,2,0)), mask=np.array(mask))

        return TensorImage(aug['image'].transpose(2,0,1)), TensorMask(aug['mask'])



# Now we will create a pipe of transformations

aug_pipe = A.Compose([A.ShiftScaleRotate(p=.9),

                      A.HorizontalFlip(),

                      A.RandomBrightnessContrast(contrast_limit=0.0, p=1., brightness_by_max=False)])



# Create our class with this aug_pipe

aug = SegmentationAlbumentationsTransform(aug_pipe)



# And check the results

idx = 5

aug_number = 4



# Display original and some augmented samples

_, ax = plt.subplots(aug_number+1, 2, figsize=(8,aug_number*4))



show_img(ds[idx][0], ctx=ax[0,0])

ds[idx][1].show(ctx=ax[0,1])



# print(ds[idx][0])



for i in range(1, aug_number+1):

    img, mask = aug.encodes(ds[idx])

    show_img(img, ctx=ax[i,0])

    mask.show(ctx=ax[i,1])

    

#     print(img)
db = DataBlock(blocks=(TransformBlock([get_filenames, open_ms_tif, lambda x: x/10000]), 

                       TransformBlock([partial(map_filename, str1='red', str2='gt'), 

                                       partial(open_tif, cls=TensorMask)])),

               splitter=RandomSplitter(valid_pct=0.2),

               item_tfms=aug,

              )



dl = db.dataloaders(items, bs=12)



# check if it is being applied correctly

_, ax = plt.subplots(2, 4, figsize=(16, 8))



for i in range(0, 4):

    img, mask = dl.do_item(idx)



    show_img(img, ctx=ax[0,i])

    mask.show(ctx=ax[1,i])
def acc_metric(input, target):

    target = target.squeeze(1)

    return (input.argmax(dim=1)==target).float().mean()



def loss_fn(pred, targ):

    targ[targ==255] = 1

    return torch.nn.functional.cross_entropy(pred, targ.squeeze(1).type(torch.long))



learn = unet_learner(dl, resnet18, n_in=4, n_out=2, pretrained=False, loss_func=loss_fn, metrics=acc_metric)
# At this notebook version we are starting from a previously saved checkpoint

# We will then, load the previous weights and train for 5 more epochs



# The final objective is to compare the final accuracy, with the accuracy introducing augmentations

try:

    learn.load('/kaggle/input/remotesensing-fastai2-multiband-augmentations/models/95_cloud-resnet18-50-35epochs_aug.learner')

    print('Loaded sucessfully')

    learn.fit_one_cycle(15, lr_max=1e-4, wd=1e-1)

    learn.save('./95_cloud-resnet18-50-35epochs_aug.learner')

except:

    

    print('failed loading checkpoint')

    



# learn.lr_find()
# learn.lr_find()