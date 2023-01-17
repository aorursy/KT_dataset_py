!pip install catalyst

!pip install pretrainedmodels

!pip install git+https://github.com/qubvel/segmentation_models.pytorch

!pip install pytorch_toolbelt

!pip install torchvision==0.4

!pip install albumentations==0.3.2

!pip install mlcomp
!ls ../input
import os

import cv2

import collections

import time 

import tqdm

from PIL import Image

from functools import partial

train_on_gpu = True



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



import torchvision

import torchvision.transforms as transforms

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR



import albumentations as albu

from albumentations import torch as AT



from catalyst.data import Augmentor

from catalyst.dl import utils

from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader

from catalyst.dl.runner import SupervisedRunner

from catalyst.contrib.models.segmentation import Unet

from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback



import segmentation_models_pytorch as smp

from tqdm import tqdm, tqdm_notebook



import warnings

warnings.filterwarnings('ignore')



from torch.jit import load



from mlcomp.contrib.transform.albumentations import ChannelTranspose

from mlcomp.contrib.dataset.classify import ImageDataset

from mlcomp.contrib.transform.rle import rle2mask, mask2rle

from mlcomp.contrib.transform.tta import TtaWrap
def get_img(x, folder: str='train_images'):

    """

    Return image based on image name and folder.

    """

    data_folder = f"{path}/{folder}"

    image_path = os.path.join(data_folder, x)

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img





def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):

    '''

    Decode rle encoded mask.

    

    :param mask_rle: run-length as string formatted (start length)

    :param shape: (height, width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape, order='F')





def make_mask(df: pd.DataFrame, image_name: str='img.jpg', shape: tuple = (1400, 2100)):

    """

    Create mask based on df, image name and shape.

    """

    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']

    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)



    for idx, label in enumerate(encoded_masks.values):

        if label is not np.nan:

            mask = rle_decode(label)

            masks[:, :, idx] = mask

            

    return masks





def to_tensor(x, **kwargs):

    """

    Convert image or mask.

    """

    return x.transpose(2, 0, 1).astype('float32')





def mask2rle(img):

    '''

    Convert mask to rle.

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)





def visualize(image, mask, original_image=None, original_mask=None):

    """

    Plot image and masks.

    If two pairs of images and masks are passes, show both.

    """

    fontsize = 14

    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    

    if original_image is None and original_mask is None:

        f, ax = plt.subplots(1, 5, figsize=(24, 24))



        ax[0].imshow(image)

        for i in range(4):

            ax[i + 1].imshow(mask[:, :, i])

            ax[i + 1].set_title(f'Mask {class_dict[i]}', fontsize=fontsize)

    else:

        f, ax = plt.subplots(2, 5, figsize=(24, 12))



        ax[0, 0].imshow(original_image)

        ax[0, 0].set_title('Original image', fontsize=fontsize)

                

        for i in range(4):

            ax[0, i + 1].imshow(original_mask[:, :, i])

            ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)

        

        ax[1, 0].imshow(image)

        ax[1, 0].set_title('Transformed image', fontsize=fontsize)

        

        

        for i in range(4):

            ax[1, i + 1].imshow(mask[:, :, i])

            ax[1, i + 1].set_title(f'Transformed mask {class_dict[i]}', fontsize=fontsize)

            

            

def visualize_with_raw(image, mask, original_image=None, original_mask=None, raw_image=None, raw_mask=None):

    """

    Plot image and masks.

    If two pairs of images and masks are passes, show both.

    """

    fontsize = 14

    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}



    f, ax = plt.subplots(3, 5, figsize=(24, 12))



    ax[0, 0].imshow(original_image)

    ax[0, 0].set_title('Original image', fontsize=fontsize)



    for i in range(4):

        ax[0, i + 1].imshow(original_mask[:, :, i])

        ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)





    ax[1, 0].imshow(raw_image)

    ax[1, 0].set_title('Original image', fontsize=fontsize)



    for i in range(4):

        ax[1, i + 1].imshow(raw_mask[:, :, i])

        ax[1, i + 1].set_title(f'Raw predicted mask {class_dict[i]}', fontsize=fontsize)

        

    ax[2, 0].imshow(image)

    ax[2, 0].set_title('Transformed image', fontsize=fontsize)





    for i in range(4):

        ax[2, i + 1].imshow(mask[:, :, i])

        ax[2, i + 1].set_title(f'Predicted mask with processing {class_dict[i]}', fontsize=fontsize)

            

            

def plot_with_augmentation(image, mask, augment):

    """

    Wrapper for `visualize` function.

    """

    augmented = augment(image=image, mask=mask)

    image_flipped = augmented['image']

    mask_flipped = augmented['mask']

    visualize(image_flipped, mask_flipped, original_image=image, original_mask=mask)

    

    

sigmoid = lambda x: 1 / (1 + np.exp(-x))





def post_process(probability, threshold, min_size):

    """

    Post processing of each predicted mask, components with lesser number of pixels

    than `min_size` are ignored

    """

    # don't remember where I saw it

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((350, 525), np.float32)

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num





def get_training_augmentation():

    train_transform = [

        albu.Resize(704, 1056),

        albu.HorizontalFlip(p=0.5),

        albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),

        # albu.GridDistortion(p=0.2),

        # albu.OpticalDistortion(p=0.2, distort_limit=2, shift_limit=0.5),

    ]

    return albu.Compose(train_transform)





def get_validation_augmentation():

    """Add paddings to make image shape divisible by 32"""

    test_transform = [

        albu.Resize(704, 1056)

    ]

    return albu.Compose(test_transform)





def get_preprocessing(preprocessing_fn):

    """Construct preprocessing transform

    

    Args:

        preprocessing_fn (callbale): data normalization function 

            (can be specific for each pretrained neural network)

    Return:

        transform: albumentations.Compose

    

    """

    

    _transform = [

        albu.Lambda(image=preprocessing_fn),

        albu.Lambda(image=to_tensor, mask=to_tensor),

    ]

    return albu.Compose(_transform)





def dice(img1, img2):

    img1 = np.asarray(img1).astype(np.bool)

    img2 = np.asarray(img2).astype(np.bool)



    intersection = np.logical_and(img1, img2)



    return 2. * intersection.sum() / (img1.sum() + img2.sum())





def ensemble_voting(p_channel, threshold, min_size):

    mask = cv2.threshold(p_channel, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((350, 525), np.float32)

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER = 'resnet50'

ENCODER_WEIGHTS = 'imagenet'



ACTIVATION = None

unet_resnet50 = smp.Unet(

    encoder_name=ENCODER, 

    encoder_weights=ENCODER_WEIGHTS, 

    classes=4, 

    activation=ACTIVATION,

)
ENCODER = 'resnet18'

ENCODER_WEIGHTS = 'imagenet'



ACTIVATION = None

unet_resnet18 = smp.Unet(

    encoder_name=ENCODER, 

    encoder_weights=ENCODER_WEIGHTS, 

    classes=4, 

    activation=ACTIVATION,

)
model_meta = torch.load('../input/segmentation-in-pytorch-using-convenient-tools/logs/segmentation/checkpoints/best.pth')

unet_resnet50.load_state_dict(model_meta["model_state_dict"])



model_meta = torch.load('../input/turbo-charging-andrew-s-pytorch/logs/segmentation_unet/checkpoints/best.pth')

unet_resnet18.load_state_dict(model_meta["model_state_dict"])
unet_resnet50.to(DEVICE);

unet_resnet18.to(DEVICE);



unet_resnet50.eval();

unet_resnet18.eval();
class Model:

    def __init__(self, models, voting=False):

        self.models = models

        self.voting = voting

    

    def __call__(self, x):

        res = []

        x = x.to(DEVICE)

        with torch.no_grad():

            for m in self.models:

                res.append(m(x))

        res = torch.stack(res)

        if (self.voting):

            return res

        return torch.mean(res, dim=0)
models = [unet_resnet50, unet_resnet18, unet_resnet18]
path = '../input/understanding_cloud_organization'



def create_transforms(additional):

    res = list(additional)

    # add necessary transformations

    res.extend([

        albu.Normalize(

            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

        ),

        ChannelTranspose()

    ])

    res = albu.Compose(res)

    return res



img_folder = f'{path}/test_images'

batch_size = 2

num_workers = 0



# Different transforms for TTA wrapper

transforms = [

    [albu.Resize(352, 576)],

    [albu.Resize(352, 576), albu.HorizontalFlip(p=1.0)],

    [albu.Resize(352, 576), albu.VerticalFlip(p=1.0)]

]



transforms = [create_transforms(t) for t in transforms]

datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]

loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]
# set ensemble mode to simple average

model = Model(models, voting=False)
# I chose the thresholds randomly, but a better way would be to decide based on validation set



thresholds = [0.5, 0.5, 0.5, 0.5]

min_area = [20000, 20000, 20000, 15000]



class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

count_flag = 0 # Jung, counting how many we have fixed





res = []

# Iterate over all TTA loaders

total = len(datasets[0])//batch_size

for loaders_batch in tqdm_notebook(zip(*loaders), total=total):

    preds = []

    # preds_sig = []

    image_file = []

    for i, batch in enumerate(loaders_batch):

        features = batch['features'].cuda()

        p = torch.sigmoid(model(features))

        # inverse operations for TTA

        p = datasets[i].inverse(p)

        preds.append(p)

        image_file = batch['image_file']

    

    # TTA mean

    preds = torch.stack(preds)

    preds = torch.mean(preds, dim=0)

    preds = preds.detach().cpu().numpy()

    

    # Batch post processing

    for p, file in zip(preds, image_file):

        file = os.path.basename(file)

        # Image postprocessing

        for i in range(4):

            imageid_classid = file + '_' + str(class_dict[i])

            p_channel = p[i]

            if p_channel.shape != (350, 525):

                p_channel = cv2.resize(p_channel, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)



            predict, num_predict = post_process(p_channel, thresholds[i], min_area[i])

            rle_mask = ''

            if num_predict != 0:

                rle_mask = mask2rle(predict)

                flag=True

            res.append({

                'Image_Label': imageid_classid,

                'EncodedPixels': rle_mask

            })
image_vis = features[0].detach().cpu().numpy().transpose(1, 2, 0)

mask = preds[0].astype('uint8').transpose(1, 2, 0)

pr_mask = np.zeros((350, 525, 4))

for j in range(4):

    probability = cv2.resize(preds[0].transpose(1, 2, 0)[:, :, j], dsize=(525, 350), interpolation=cv2.INTER_LINEAR)

    pr_mask[:, :, j], _ = post_process(probability, thresholds[j], min_area[j])



visualize_with_raw(image=image_vis, mask=pr_mask, original_image=image_vis, original_mask=mask, raw_image=image_vis, raw_mask=preds[0].transpose(1, 2, 0))
df = pd.DataFrame(res)

df.to_csv('submission_avg_ensemble.csv', index=False)
# set ensemble mode to voting

model_voting = Model(models, voting=True)
thresholds = [0.5, 0.5, 0.5, 0.5]

min_area = [20000, 20000, 20000, 15000]



class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}



res = []

# Iterate over all TTA loaders

total = len(datasets[0])//batch_size

for loaders_batch in tqdm_notebook(zip(*loaders), total=total):

    preds = []

    # preds_sig = []

    image_file = []

    for i, batch in enumerate(loaders_batch):

        features = batch['features'].cuda()

        p = torch.sigmoid(model_voting(features))

        # inverse operations for TTA

        p = datasets[i].inverse(p)

        preds.append(p)

        image_file = batch['image_file']

    

    # TTA mean

    preds = torch.stack(preds)

    preds = torch.mean(preds, dim=0)

    preds = preds.detach().cpu().numpy()

    preds = np.transpose(preds, [1, 0, 2, 3, 4])

    

    # Batch post processing

    for p, file in zip(preds, image_file):

        file = os.path.basename(file)

        # for all models

        for i in range(4):

            # Image postprocessing

            pred_masks = []

            for m in range(len(models)):

                imageid_classid = file + '_' + str(class_dict[i])

                p_channel = p[m][i]

                if p_channel.shape != (350, 525):

                    p_channel = cv2.resize(p_channel, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)



                mask = cv2.threshold(p_channel, thresholds[i], 1, cv2.THRESH_BINARY)[1]

                pred_masks.append(torch.from_numpy(mask))



            voting_mask = torch.mean(torch.stack(pred_masks), dim=0)

            

            # based on number of models choose the ensemble voting threshold, in my case i have 3 models if any 2 vote 

            # for 1 pixel we choose that pixel in answer

            predict, num_predict = ensemble_voting(voting_mask.numpy(), 0.5, min_area[i])

          

            rle_mask = ''

            if num_predict != 0:

                rle_mask = mask2rle(predict)



            res.append({

              'Image_Label': imageid_classid,

              'EncodedPixels': rle_mask

            })
image_vis = features[0].detach().cpu().numpy().transpose(1, 2, 0)

maskk = preds[0].astype('uint8').transpose(0, 2, 3, 1)

pr_mask = np.zeros((350, 525, 4))

for i in range(4):

    # Image postprocessing

    pred_masks = []

    for m in range(len(models)):

        p_channel = preds[0][m][i]

        if p_channel.shape != (350, 525):

            p_channel = cv2.resize(p_channel, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)



      # predict, num_predict = post_process(p_channel, thresholds[i], min_area[i])

        mask = cv2.threshold(p_channel, thresholds[i], 1, cv2.THRESH_BINARY)[1]

        pred_masks.append(torch.from_numpy(mask))



    voting_mask = torch.mean(torch.stack(pred_masks), dim=0)

    pr_mask[:, :, i], _ = ensemble_voting(voting_mask.numpy(), 0.5, min_area[i])





visualize_with_raw(image=image_vis, mask=pr_mask, original_image=image_vis,

                   original_mask=np.mean(maskk, axis=0), raw_image=image_vis, raw_mask=np.mean(preds[0], axis=0).transpose(1, 2, 0))
df = pd.DataFrame(res)

df.to_csv('submission_voting_ensemble.csv', index=False)