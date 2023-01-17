seed = 42



# python RNG

import random

random.seed(seed)



# pytorch RNGs

import torch

torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True

if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



# numpy RNG

import numpy as np

np.random.seed(seed)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

##

# Any results you write to the current directory are saved as output.



# Ignoring the warnings

import warnings

import pdb

from pathlib import Path



from fastai.vision import *

from fastai.callbacks import *



from fastai.utils.collect_env import *



warnings.filterwarnings("ignore")

show_install(True)



# Path to the data directory which contains all the data.

data_dir = Path('/kaggle/input/openai-constr-starts-foundation/openAI_constr_starts_foundation')

#  /kaggle/input/openai-constr-starts-foundation/openAI_constr_starts_foundation/images

# Path to the models directory which contains all the models.

model_dir = '/kaggle/working'



training_images_path = data_dir/'images/'

training_masks_path = data_dir/'masks/'



file_names = get_image_files(training_images_path, recurse=True)

masks_names = get_image_files(training_masks_path, recurse=True)



print('The total number of training images: ', len(file_names))

print('The total number of training masks: ', len(masks_names))



print(file_names[:3], masks_names[:3])





def get_y_fn(image_path):

    #return training_masks_path/f'{image_path.stem}_mask.png'

    return training_masks_path/Path(str(image_path.relative_to(Path(training_images_path)).parent).replace('images','masks'))/f'{image_path.stem}_mask.png'



# test that masks are opening correctly with open_mask() settings

sample_image_path = file_names[166]

sample_image = open_image(sample_image_path)

sample_mask = open_mask(get_y_fn(sample_image_path), convert_mode='RGB', div=False)



fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sample_image.show(ax=ax)

sample_mask.show(ax=ax, alpha=0.5)

plt.show()



plt.hist(sample_mask.data.view(-1), bins=3)



# define the validation set by fn prefix

holdout_grids = ['znz001val_']

valid_idx = [i for i, o in enumerate(file_names) if any(c in str(o) for c in holdout_grids)]

print('The total number of validation data is: ', len(valid_idx))





# Subclassing SegmentationLabelList to set open_mask(fn, div=True, convert_mode='RGB') for 3 channel target masks



class SegLabelListCustom(SegmentationLabelList):

    @staticmethod

    # Modifies open function of SegmentationLabelList to open our mask images.

    def open(fn):

        # creates imageSegment object, normalize masks pixel by 255! Expect 0-255 masks

        return open_mask(fn, div=True, convert_mode='RGB')





class SegItemListCustom(SegmentationItemList):

    # SegmentationItemList is subclass of ItemList for segmentation

    # modifies _label_cls which holds subclass of SegmentationLabelList

    _label_cls = SegLabelListCustom





# The classes corresponding to each channel

codes = np.array(['Footprint', 'Boundary', 'Contact'])



size = 256  # Tile-size

batch_size = 16



# Define image transforms for data augmentation and create databunch. More about image tfms and data aug at

# Image augmentation: 

# * random flip with p=0.5, vertically and horizontal flip

# * rotation between -20 and 20 degrees with p=0.75

# * random symmetric warp of magnitude between -max_warp and maw_warp with p=0.75

# * random lightning and contrast change controlled by max_lighting=0.3

# * zoom between 1 and max_zoom with p=0.75

tfms = get_transforms(flip_vert=True, max_warp=0.1, max_rotate=20, max_zoom=2, max_lighting=0.3)



# leveragin the customizable data block API

src = (SegItemListCustom.from_folder(training_images_path)

       .split_by_idx(valid_idx)

       .label_from_func(get_y_fn, classes=codes))



# tfm_y applies same transformation to masks

# create databunch, normalize with imagenet mean, std

data = (src.transform(tfms, size=size, tfm_y=True)

        .databunch(bs=batch_size, num_workers=1)

        .normalize(imagenet_stats))





def show_3ch(imgitem, figsize=(10, 5)):

    # displaying the 3 channels of the image.

    figure, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=figsize)

    axis1.imshow(np.asarray(imgitem.data[0, None])[0])

    axis2.imshow(np.asarray(imgitem.data[1, None])[0])

    axis3.imshow(np.asarray(imgitem.data[2, None])[0])



    axis1.set_title('Footprint')

    axis2.set_title('Boundary')

    axis3.set_title('Contact')



    plt.show()





for idx in range(10, 15):

    print(data.valid_ds.items[idx].name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    data.valid_ds.x[idx].show(ax=ax1)

    ax2.imshow(image2np(data.valid_ds.y[idx].data*255))

    sample_image.show(ax=ax)

    plt.show()

    show_3ch(data.valid_ds.y[idx])



# Visually inspect data-augmented training images

# TODO: show_batch doesn't display RGB mask correctly, setting alpha=0 to turn off for now

data.show_batch(4, figsize=(10, 10), alpha=0.)



print('Details about the data: ', data)





# Define custom losses and metrics to handle 3-channel targets

def dice_loss(input_image, target):

    # pdb.set_trace()

    smooth = 1.

    input_image = torch.sigmoid(input_image)

    iflat = input_image.contiguous().view(-1).float()

    tflat = target.contiguous().view(-1).float()

    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / ((iflat + tflat).sum() + smooth))





# Adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938

class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction='mean'):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.reduction = reduction



    def forward(self, inputs, targets):

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')

        pt = torch.exp(-bce_loss)

        f_loss = self.alpha * (1-pt)**self.gamma * bce_loss



        if self.reduction == 'mean':

            return f_loss.mean()

        elif self.reduction == 'sum':

            return f_loss.sum()

        else:

            return f_loss





class DiceLoss(nn.Module):

    def __init__(self, reduction='mean'):

        super().__init__()

        self.reduction = reduction



    def forward(self, input_image, target):

        loss = dice_loss(input_image, target)

        if self.reduction == 'mean':

            return loss.mean()

        elif self.reduction == 'sum':

            return loss.sum()

        else:

            return loss





class MultiChComboLoss(nn.Module):

    def __init__(self, reduction='mean', loss_funcs=[FocalLoss(), DiceLoss()], loss_wts=[1, 1], ch_wts=[1, 1, 1]):

        super().__init__()

        self.reduction = reduction

        self.ch_wts = ch_wts

        self.loss_wts = loss_wts

        self.loss_funcs = loss_funcs



    def forward(self, output, target):

        # pdb.set_trace()

        # need to change reduction on fwd pass for loss calc in learn.get_preds(with_loss=True)

        for loss_func in self.loss_funcs:

            loss_func.reduction = self.reduction

        loss = 0

        channels = output.shape[1]

        assert len(self.ch_wts) == channels

        assert len(self.loss_wts) == len(self.loss_funcs)

        for ch_wt, c in zip(self.ch_wts, range(channels)):

            ch_loss = 0

            for loss_wt, loss_func in zip(self.loss_wts, self.loss_funcs):

                ch_loss += loss_wt * loss_func(output[:, c, None], target[:, c, None])

            loss += ch_wt * ch_loss

        return loss/sum(self.ch_wts)





# Calculate metrics on one channel (i.e. ch 0 for building footprints only) or on all 3 channels

def acc_thresh_multich(input_image: Tensor, target: Tensor, thresh: float = 0.5,

                       sigmoid: bool = True, one_ch: int = None) -> Rank0Tensor:



    """Compute accuracy when `y_pred` and `y_true` are the same size."""



    # pdb.set_trace()

    if sigmoid:

        input_image = input_image.sigmoid()

    n = input_image.shape[0]



    if one_ch is not None:

        input_image = input_image[:, one_ch, None]

        target = target[:, one_ch, None]



    input_image = input_image.view(n, -1)

    target = target.view(n, -1)

    return ((input_image > thresh) == target.byte()).float().mean()





def dice_multich(input_image: Tensor, targs: Tensor, iou: bool = False, one_ch: int = None) -> Rank0Tensor:

    """Dice coefficient metric for binary target. If iou=True, returns iou metric, classic or segmentation problems."""

    # pdb.set_trace()

    n = targs.shape[0]

    input_image = input_image.sigmoid()



    if one_ch is not None:

        input_image = input_image[:, one_ch, None]

        targs = targs[:, one_ch, None]



    input_image = (input_image > 0.5).view(n, -1).float()

    targs = targs.view(n, -1).float()



    intersect = (input_image * targs).sum().float()

    union = (input_image + targs).sum().float()

    if not iou:

        return 2. * intersect / union if union > 0 else union.new([1.]).squeeze()

    else:

        return intersect / (union-intersect+1.0)

#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(filename)
# Set up metrics to show mean metrics for all channels as well as the building-only metrics (channel 0)

acc_ch0 = partial(acc_thresh_multich, one_ch=0)

dice_ch0 = partial(dice_multich, one_ch=0)

metrics = [acc_thresh_multich, dice_multich, acc_ch0, dice_ch0]



# Combo Focal + Dice loss with equal channel wts

learn = unet_learner(data, models.resnet34, model_dir=model_dir, metrics=metrics, loss_func=MultiChComboLoss(reduction='mean',

                         loss_funcs=[FocalLoss(gamma=1, alpha=0.95),

                                     DiceLoss(),

                                     ],

                         loss_wts=[1, 1],

                         ch_wts=[1, 1, 1]))

tmp = learn.load("../input/constrstarts1-buildings/buildings-focaldice-unfrozen-best")

learn.freeze()
print('The metrics for U-net learner are: ', learn.metrics)

print('The loss function for the U-net learner is: ', learn.loss_func)

#print('U-net learner summary: \n', learn.summary())
learn.lr_find()
learn.recorder.plot(0, 2, suggestion=True)
learning_rate = 1.5e-3 #original LR

#learning_rate = 1e-4 #2.e-3
learn.fit_one_cycle(10, max_lr=learning_rate,

                    callbacks=[

                        SaveModelCallback(learn,

                                          monitor='dice_multich',

                                          mode='max',

                                          name='foundationsExp1-focaldice-stage1-best')

                    ]

                    )
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
def compare_image_results(dataset,outputs, losses_reshaped,sorted_idx,labels,i):

#for i in sorted_idx[:25]: # plot 10 images

    print(f'{dataset.items[i].name}')

    print(f'loss: {losses_reshaped[i].mean()}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))



    dataset.x[i].show(ax=ax1)

    ax1.set_title('Prediction')

    ax1.imshow(image2np(outputs[i].sigmoid()), alpha=0.4)



    ax2.set_title('Ground Truth')

    dataset.x[i].show(ax=ax2)

    ax2.imshow(image2np(labels[i])*255, alpha=0.4)



    print('Predicted:')

    show_3ch(outputs[i].sigmoid())

    print('Actual:')

    show_3ch(labels[i])
outputs, labels, losses = learn.get_preds(ds_type=DatasetType.Fix, n_batch=3, with_loss=True)

print(outputs.shape,labels.shape)

print('Dimensions of the loss: ', losses.shape)



losses_reshaped = torch.mean(losses.view(outputs.shape[0], -1), dim=1)

sorted_idx = torch.argsort(losses_reshaped, descending=True) #identify ...

print('Dimensions of the reshaped losses: ', losses_reshaped.shape)



for idx in sorted_idx[:25]: # plot 10 images

    compare_image_results(data.train_ds,outputs,losses_reshaped,sorted_idx,labels,idx)
learn.model.eval()

# run prediction on 3 batches (16 datapoints) all validation data

outputs, labels, losses = learn.get_preds(ds_type=DatasetType.Valid, n_batch=3, with_loss=True)

print(outputs.shape,labels.shape)

print('Dimensions of the loss: ', losses.shape)



losses_reshaped = torch.mean(losses.view(outputs.shape[0], -1), dim=1)

sorted_idx = torch.argsort(losses_reshaped, descending=True) #identify ...

print('Dimensions of the reshaped losses: ', losses_reshaped.shape)



# look at predictions vs actual by channel sorted by highest image-wise loss first



for idx in sorted_idx[:25]: # plot 10 images

    compare_image_results(data.valid_ds,outputs,losses_reshaped,sorted_idx,labels,idx)

    
tmp_output = learn.load('foundationsExp1-focaldice-stage1-best')
learn.model.train()

learn.unfreeze() # unfreeze the entire model and every layer becomes trainable
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(20, max_lr=slice(3e-6, 3e-4),

                    callbacks=[

                        SaveModelCallback(learn,

                                          monitor='dice_multich',

                                          mode='max',

                                          name='foundationsExp1-focaldice-unfrozen-best')

                    ]

                    )
learn.recorder.plot_losses() #; plt.ylim(0.36,0.44) #adjust limits if needed
learn.recorder.plot_metrics()
outputs, labels, losses = learn.get_preds(ds_type=DatasetType.Fix, n_batch=3, with_loss=True)

print(outputs.shape,labels.shape)

print('Dimensions of the loss: ', losses.shape)



losses_reshaped = torch.mean(losses.view(outputs.shape[0], -1), dim=1)

sorted_idx = torch.argsort(losses_reshaped, descending=True) #identify ...

print('Dimensions of the reshaped losses: ', losses_reshaped.shape)



for idx in sorted_idx[:25]: # plot 10 images

    compare_image_results(data.train_ds,outputs,losses_reshaped,sorted_idx,labels,idx)
# run prediction on 3 batches (16 datapoints) all validation data

outputs, labels, losses = learn.get_preds(ds_type=DatasetType.Valid, n_batch=3, with_loss=True)

print(outputs.shape,labels.shape)

print('Dimensions of the loss: ', losses.shape)



losses_reshaped = torch.mean(losses.view(outputs.shape[0], -1), dim=1)

sorted_idx = torch.argsort(losses_reshaped, descending=True) #identify ...

print('Dimensions of the reshaped losses: ', losses_reshaped.shape)



# look at predictions vs actual by channel sorted by highest image-wise loss first



for idx in sorted_idx[:25]: # plot 10 images

    compare_image_results(data.valid_ds,outputs,losses_reshaped,sorted_idx,labels,idx)
learn.metrics
learn.validate(data.train_dl)
learn.validate(data.valid_dl)
learn.export('/kaggle/working/foundationsExp1-focaldice-export-best.pkl')