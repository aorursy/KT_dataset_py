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



# Any results you write to the current directory are saved as output.
# Imports

import openslide

import PIL

import skimage.io

import cv2

from tqdm.notebook import tqdm

import time

import torch

import sys

!pip install pretrainedmodels

import pretrainedmodels

from sklearn import model_selection

import torch.nn as nn

import albumentations
# PATHS

IMG_PATH = '../input/prostate-cancer-grade-assessment/train_images/'

MSK_PATH = '../input/prostate-cancer-grade-assessment/train_label_masks/'

SAVE_PATH = '/kaggle/Resized/'

TEST_PATH = '../input/prostate-cancer-grade-assessment/test_images/'



DIMS = (512, 512)

TRAIN_BAT_SIZE = 32

VALID_BAT_SIZE = 16

DEVICE = 'cuda'

EPOCHS = 40

FOLDS = 4

TRAIN_FOLDS = [4,3,1,2] # FOLDS must always remain in train folds for StratifiedShuffleSplit

VAL_FOLDS = [0]

MODEL_MEAN=(0.485, 0.456, 0.406)

MODEL_STD=(0.229, 0.224, 0.225)



!ls ../input/prostate-cancer-grade-assessment
train_csv = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

# train_csv = train_csv.sample(frac = 1).reset_index(drop=True)



train_csv['gleason_score'] = train_csv['gleason_score'].replace('negative', '0+0')

train_csv['g_score1'] = train_csv['gleason_score'].apply(lambda x: int(x.split('+')[0]))

train_csv['g_score2'] = train_csv['gleason_score'].apply(lambda x: int(x.split('+')[1]))



train_csv['kfolds'] = FOLDS

# kf = model_selection.StratifiedKFold(n_splits = FOLDS, shuffle = False, random_state = 10)

sss = model_selection.StratifiedShuffleSplit(n_splits=FOLDS, test_size=0.05, random_state = 10)

for fold, (train_idx, val_idx) in enumerate(sss.split(X = train_csv, y=train_csv.isup_grade.values)):

    print(len(train_idx), len(val_idx))

    train_csv.loc[val_idx, 'kfolds'] = fold

# train_csv.head()

train_csv.to_csv('/kaggle/train_folds.csv', index = False)

train_csv = pd.read_csv('/kaggle/train_folds.csv')

train_csv['has_mask'] = 1

# train_csv[train_csv.kfolds == 0]

# train_csv.kfolds.unique()
def print_slide_details(slide, show_thumbnail=True, max_size=(600,400)):

    """Print some basic information about a slide"""

    # Generate a small image thumbnail

    if show_thumbnail:

        display(slide.get_thumbnail(size=max_size))



    # Here we compute the "pixel spacing": the physical size of a pixel in the image.

    # OpenSlide gives the resolution in centimeters so we convert this to microns.

    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)

    

    print(f"File id: {slide}")

    print(f"Dimensions: {slide.dimensions}")

    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")

    print(f"Number of levels in the image: {slide.level_count}")

    print(f"Downsample factor per level: {slide.level_downsamples}")

    print(f"Dimensions of levels: {slide.level_dimensions}")
# example_slides = [

#     '005e66f06bce9c2e49142536caf2f6ee',

#     '00928370e2dfeb8a507667ef1d4efcbb',

#     '007433133235efc27a39f11df6940829',

#     '024ed1244a6d817358cedaea3783bbde',

# ]



# for case_id in example_slides:

#     biopsy = openslide.OpenSlide(os.path.join(IMG_PATH, f'{case_id}.tiff'))

#     print_slide_details(biopsy)

#     biopsy.close()
os.makedirs(SAVE_PATH, exist_ok = True)

!ls
def resize_and_save(img_id, dim, level):

    img = skimage.io.MultiImage(IMG_PATH+img_id+'.tiff')

    out = cv2.resize(img[level], dim)

    cv2.imwrite(SAVE_PATH+f'{img_id}.png',out)

    

def mask_resize_and_save(img_id, dim, level):

    try:

        img = skimage.io.MultiImage(MSK_PATH+img_id+'_mask.tiff')

        out = cv2.resize(img[level], dim)

        cv2.imwrite(SAVE_PATH+f'{img_id}_mask.png',out)

    except:

        print('hello')
# for img_id in tqdm(train_csv.image_id[:10]):

#     resize_and_save(img_id, DIMS, -1)

#     mask_resize_and_save(img_id, DIMS, -1)
# !ls /kaggle/Resized/

!ls /kaggle/Resized -1 | wc -l
def overlay_mask_on_slide(img_id, center='radboud', level = 2, alpha=0.8, max_size=(1024, 1024), view = False, mask = False):

    """Outputs Mask overlayed on a slide to a image of desired dimensions without changing the aspect ratio.

        Edited from https://www.kaggle.com/wouterbulten/getting-started-with-the-panda-dataset"""

    if not mask:

        slide = openslide.OpenSlide(os.path.join(IMG_PATH, f'{img_id}.tiff'))

        slide_data = slide.read_region((0,0), level, slide.level_dimensions[level])

        #slidPIL.Image.fromarray(slide_data).convert(mode = 'RGBA')

        background = PIL.Image.new('RGBA', max_size, (255, 255, 255, 255))

        #paste the image on max_size background

        background.paste(slide_data, (0, 0), slide_data)

        background.thumbnail(size=max_size, resample=0)

        background.save(SAVE_PATH+f'{img_id}_overlayed.png')

        

    if center not in ['radboud', 'karolinska']:

        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")



    try:

        mask = openslide.OpenSlide(os.path.join(MSK_PATH, f'{img_id}_mask.tiff'))

    except:

        # Skip over the image if the mask of image is not available

        train_csv.loc[train_csv.image_id == img_id,'has_mask'] = 0

        return

    slide = openslide.OpenSlide(os.path.join(IMG_PATH, f'{img_id}.tiff'))



    # Load data from the desired level

    slide_data = slide.read_region((0,0), level, slide.level_dimensions[level])

    mask_data = mask.read_region((0,0), level, mask.level_dimensions[level])



    # Mask data is present in the R channel

    mask_data = mask_data.split()[0]



    # Create alpha mask

    alpha_int = int(round(255*alpha))

    if center == 'radboud':

        alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)

    elif center == 'karolinska':

        alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)



    alpha_content = PIL.Image.fromarray(alpha_content)

    preview_palette = np.zeros(shape=768, dtype=int)



    if center == 'radboud':

        # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

        preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

    elif center == 'karolinska':

        # Mapping: {0: background, 1: benign, 2: cancer}

        preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)



    mask_data.putpalette(data=preview_palette.tolist())

    mask_rgb = mask_data.convert(mode='RGB')



    overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content).convert('RGBA')



    # reduce the size of image to max_size

    overlayed_image.thumbnail(size=max_size, resample=0)



    # create a white background of desired dimensions

    background = PIL.Image.new('RGBA', max_size, (255, 255, 255, 255))

    #paste the image on max_size background

    background.paste(overlayed_image, (0, 0), overlayed_image)

    background.thumbnail(size=max_size, resample=0)

    background.save(SAVE_PATH+f'{img_id}_overlayed.png')

    # To see the output images

    if view:

        display(background)

    slide.close()

    mask.close()
for i, row in tqdm(train_csv.iterrows(), total = len(train_csv)):

    overlay_mask_on_slide(row['image_id'], row['data_provider'], max_size = DIMS, mask = False)

#     print(row)
# Remove the data of images which have no mask

print(len(train_csv))

train_csv = train_csv[train_csv['has_mask'] == 1]

train_csv.to_csv('/kaggle/train_folds.csv', index = False)

len(train_csv)
# Dataset

class PANDADataset:

    def __init__(self, folds, mean, std): #, img_ht, img_wd,

        df = pd.read_csv('/kaggle/train_folds.csv')

        df = df[['image_id', 'g_score1', 'g_score2']]

#         print('ds')

#         df = df[df.kfold.isin(folds)].reset_index(drop=True)

        self.image_ids = df.image_id.values

        self.g_score1 = df.g_score1.values

        self.g_score2 = df.g_score2.values

#         self.isup = df.isup.values

        

        if len(folds) == 1:

            self.aug = albumentations.Compose([

#                 albumentations.Resize(img_ht, img_wd, always_apply = True), #  already resized

                albumentations.Normalize(mean, std, always_apply = True)

            ])

        else:

            self.aug = albumentations.Compose([

#                 albumentations.Resize(img_ht, img_wd, always_apply = True),

                albumentations.ShiftScaleRotate(shift_limit = -0.0625, 

                                                scale_limit = 0.1, 

                                                rotate_limit = 5,

                                                p = 0.9),

                albumentations.Normalize(mean, std, always_apply = True)

            ])



    def __len__(self):

        return len(self.image_ids)



    def __getitem__(self, item):

#         print('pdb')

#         import pdb; pdb.set_trace()

        image = PIL.Image.open(SAVE_PATH+self.image_ids[item]+'_overlayed.png').convert('RGB')

#         image = image.reshape(137, 236).astype(float)

#         image = Image.fromarray(image).convert('RGB')

        image = self.aug(image = np.array(image))['image']

        # pdb.set_trace()

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {

            'image': torch.tensor(image, dtype = torch.float),

            'g_score1': torch.tensor(self.g_score1[item], dtype = torch.long),

            'g_score2': torch.tensor(self.g_score2[item], dtype = torch.long),

#             'isup': torch.tensor(self.isup[item], dtype = torch.long),

        }
# Model 1'Resnet34'



class ResNet34(nn.Module):

    def __init__(self, pretrained, freeze = True):

        super(ResNet34, self).__init__()

        if pretrained:

            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = 'imagenet')

        else:

            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = None)

        if freeze:

            for param in self.model.parameters():

                param.requires_grad = False

        layers = []

        layers.append(nn.Linear(512, 256))

        layers.append(nn.ReLU())

        layers.append(nn.Linear(256, 6))

        layers.append(nn.ReLU())

        layers.append(nn.Linear(6, 128))

        layers.append(nn.ReLU())

        layers.append(nn.Linear(128, 256))

        layers.append(nn.ReLU())

        layers.append(nn.Linear(256, 6))

        layers.append(nn.Sigmoid())



        self.l0 = nn.Sequential(*layers)

        self.l1 = nn.Sequential(*layers)



    def forward(self, X):

        bs, _, _, _ = X.shape

        X = self.model.features(X)

        X = nn.functional.adaptive_avg_pool2d(X, 1).reshape(bs, -1)

        l0 = self.l0(X)

        l1 = self.l1(X)

        return l0, l1
# Training and Eval code



def train(dataset, data_loader, model, optimizer):

    model.train()

    for bi, d in tqdm(enumerate(data_loader), total = int(len(dataset) / data_loader.batch_size)):

#         print('d')

        image = d['image']

        g_score1 = d['g_score1']

        g_score2 = d['g_score2']

        

        image = image.to(DEVICE, dtype = torch.float)

        g_score1 = g_score1.to(DEVICE, dtype = torch.long)

        g_score2 = g_score2.to(DEVICE, dtype = torch.long)

        

        # resets gradients to zero for at the start of every mini-batch, so as not mix up the gradients.

        optimizer.zero_grad()

        

        outputs = model(image)

        targets = (g_score1, g_score2)

        

        loss = loss_function(outputs, targets)

        

        ''' Calling .backward() mutiple times accumulates the gradient (by addition) for each parameter.

            This is why you should call optimizer.zero_grad() after each .step() call. 

            Note that following the first .backward call,

            a second call is only possible after you have performed another forward pass

        '''

        loss.backward()



        '''optimizer.step is performs a parameter update based on the current gradient 

           (stored in .grad attribute of a parameter) and the update rule

        '''

        optimizer.step()



def eval(dataset, data_loader, model):

    model.eval()

    final_loss = 0

    counter = 0

    for bi, d in tqdm(enumerate(data_loader), total = int(len(dataset) / data_loader.batch_size)):

        counter += 1

        image = d['image']

        g_score1 = d['g_score1']

        g_score2 = d['g_score2']

        

        image = image.to(DEVICE, dtype = torch.float)

        g_score1 = g_score1.to(DEVICE, dtype = torch.long)

        g_score2 = g_score2.to(DEVICE, dtype = torch.long)

        

        outputs = model(image)

        targets = (g_score1, g_score2)

        loss = loss_function(outputs, targets)

        final_loss += loss

    return final_loss/counter



def loss_function(outputs, targets):

    o1, o2 = outputs

    t1, t2 = targets

    

    l1 = nn.CrossEntropyLoss()(o1, t1)

    l2 = nn.CrossEntropyLoss()(o2, t2)

    

    return (l1 + l2) / 2
# Driver Code

def main():

    model = ResNet34(pretrained = True, freeze = False)

    model.to(DEVICE)

    

    train_dataset = PANDADataset(

        folds = TRAIN_FOLDS,

#         img_ht = DIMS[0],

#         img_wd = DIMS[1],

        mean = MODEL_MEAN,

        std = MODEL_STD

    )

    

    train_loader = torch.utils.data.DataLoader(

        dataset = train_dataset,

        batch_size = TRAIN_BAT_SIZE,

        shuffle = True,

        num_workers = 4

    )

    

    valid_dataset = PANDADataset(

        folds = VAL_FOLDS,

#         img_ht = DIMS[0],

#         img_wd = DIMS[1],

        mean = MODEL_MEAN,

        std = MODEL_STD

    )

    

    valid_loader = torch.utils.data.DataLoader(

        dataset = valid_dataset,

        batch_size = VALID_BAT_SIZE,

        shuffle = True,

        num_workers = 4

    )

    

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',

                                                          patience = 3, factor = 0.3, verbose = True)

#     print('a')

    

    for epoch in tqdm(range(EPOCHS)):

#         print('b')

        train(train_dataset, train_loader, model, optimizer)

#         print('c')

        with torch.no_grad():

            val_score = eval(valid_dataset, valid_loader, model)

        scheduler.step(val_score)

        if epoch % 10 == 9:

            torch.save(model.state_dict(), f'model_{VAL_FOLDS[0]}_{epoch}.bin')
main()