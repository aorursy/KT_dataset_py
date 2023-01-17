# update torch and torch vision

!pip install -q torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# install kornia, we will give it a try to accelarate our preprocessing

!pip install -q --upgrade kornia

!pip install -q allennlp==1.1.0.rc4
# and install fastai2

!pip install -q --upgrade fastai
# Now, let's check the installed libraries



import torch

print(torch.__version__)

print(torch.cuda.is_available())



import fastai

print(fastai.__version__)



from fastai.vision.all import *
# other imports

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from PIL import Image
def open_tif(fn, chnls=None, cls=torch.Tensor):

    im = (np.array(Image.open(fn))).astype('float32')

    return cls(im)
# Just checking our function

base_path = Path('../input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/train_red_additional_to38cloud')



open_tif(base_path/'red_patch_100_5_by_16_LC08_L1TP_035031_20160120_20170224_01_T1.TIF').shape
# Note that we can pass the desired output class and it casts automatically. Here we receive a TensorImage

open_tif(base_path/'red_patch_100_5_by_16_LC08_L1TP_035031_20160120_20170224_01_T1.TIF', cls=TensorImage)
# Note that we can cast to the desired class implicitly or explicitly

mask = TensorMask(open_tif('../input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/train_gt_additional_to38cloud/gt_patch_100_5_by_16_LC08_L1TP_030034_20170815_20170825_01_T1.TIF'))

mask.show()
# get items from both datasets

items_95 = get_files(base_path, extensions='.TIF')

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

    



# Check if it is mapping correcly

for path in get_filenames(items[0]):

    assert path.exists() == True



# check if we can open an image

open_ms_tif(get_filenames(items[0])).shape
ImageBlock = TransformBlock(type_tfms=[get_filenames,

                                       open_ms_tif, 

                                       lambda x: x/10000

                                      ])
MaskBlock = TransformBlock(type_tfms=[partial(open_tif, cls=TensorMask),

                                      AddMaskCodes(codes=['clear', 'cloudy'])

                                     ])
# in this example, we don't need to specify the get_items function 

# because we will pass the list of files as the source

db = DataBlock(blocks=(ImageBlock, MaskBlock),

               get_y = partial(map_filename, str1='red', str2='gt'),

               splitter=RandomSplitter(valid_pct=0.5)

              )



# We will call db.summary to see if everything goes well

# Note that our final sample is composed by a tuple (X, Y) as we wanted

%time db.summary(source=items)
dl = db.dataloaders(source=items, bs=12) #, num_workers=0)



# now, let's check if the batch is being created correctly

batch = dl.one_batch()

batch[0].shape, batch[1].shape



# Remember that we cannot use visual funcs like show_batch as the TensorImage does not support multi channel images

# for that, we have to subclass the TensorImage as I explained in Medium:

# How to create a DataBlock for Multispectral Satellite Image Segmentation with the Fastai-v2

# https://towardsdatascience.com/how-to-create-a-datablock-for-multispectral-satellite-image-segmentation-with-the-fastai-v2-bc5e82f4eb5
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

    learn.load('/kaggle/input/remotesensing-fastai2-simpletraining/models/95_cloud-resnet18-50-10epochs.learner')

    print('Loaded sucessfully')

    learn.fit_one_cycle(5, lr_max=1e-4, wd=1e-2)

    learn.save('./95_cloud-resnet18-50-10epochs.learner')

except:

    

    print('failed loading checkpoint')

    



# learn.lr_find()
# And we save our first model to test it later
