import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
import math
import torch.utils.model_zoo as model_zoo
import cv2

import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io
import random
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold

import albumentations

# import PIL
from PIL import Image

!pip install /kaggle/input/pytorchcv/pytorchcv-0.0.55-py2.py3-none-any.whl --quiet
from pytorchcv.model_provider import get_model
import warnings
warnings.filterwarnings("ignore")
import gc


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    IMG_WIDTH = 400
    IMG_HEIGHT = 400
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    CLASSES = 6
    
seed_torch(seed=42)
N_SPLITS = 3
# Location of the training images
BASE_PATH = '../input/prostate-cancer-grade-assessment'

# image and mask directories
data_dir = f'{BASE_PATH}/train_images'
mask_dir = f'{BASE_PATH}/train_label_masks'
test_dir = f'{BASE_PATH}/test_images'
model_path = '/kaggle/input/zenify-pandas-models/'


# Location of training labels
train = pd.read_csv(f'{BASE_PATH}/train.csv')
test = pd.read_csv(f'{BASE_PATH}/test.csv')
submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
def enhance_image(image):
    img_enhanced = cv2.addWeighted(image, 1, image, 0, 15)
    return img_enhanced
    
def estimate_forground_packing_factor(image):
    """ Estimate % of forground"""
    image = enhance_image(image)
    white = len(image[np.where(image>=250)])
    non_white = len(image[np.where(image<250)])
    return 100 * (non_white/(white+non_white))

def split_image(image, img_enhanced, n_tile=4):
    """
    Splits the given image into multiple images 
    """
    
    w = int(image.shape[0]/n_tile)
    h = int(image.shape[1]/n_tile)
    
    tiles = []
    factors = []
    for i in range(n_tile):
        for j in range(n_tile):
            tile = image[int(i*w):int((i+1)*w),int(j*h):int((j+1)*h)]
            tile_enhanced = img_enhanced[int(i*w):int((i+1)*w),int(j*h):int((j+1)*h)]
            tiles.append(tile)
            factors.append(estimate_forground_packing_factor(tile_enhanced))

    return tiles, factors

def generate_tiles(img):
    img_up = remove_border(img)
    
    img_enhanced = enhance_image(img_up)
    tiles, factors = split_image(img_up, img_enhanced, 12)
    
    ind = np.argsort(factors)[::-1]

    im1 = np.concatenate((tiles[ind[0]],tiles[ind[1]],tiles[ind[2]],tiles[ind[3]],tiles[ind[4]]),axis=0)
    im2 = np.concatenate((tiles[ind[5]],tiles[ind[6]],tiles[ind[7]],tiles[ind[8]],tiles[ind[9]]),axis=0)
    im3 = np.concatenate((tiles[ind[10]],tiles[ind[11]],tiles[ind[12]],tiles[ind[13]],tiles[ind[14]]),axis=0)
    im4 = np.concatenate((tiles[ind[15]],tiles[ind[16]],tiles[ind[17]],tiles[ind[18]],tiles[ind[19]]),axis=0)
    im5 = np.concatenate((tiles[ind[20]],tiles[ind[21]],tiles[ind[22]],tiles[ind[23]],tiles[ind[24]]),axis=0)

    im_final= np.concatenate((im1,im2, im3, im4, im5),axis=1) 
    
    return im_final

def remove_border(image, mask=None):
    try:
        borders = np.where(image.sum(2) != 3*255)
        x_min = np.min(borders[0])
        x_max = np.max(borders[0]) + 1
        y_min = np.min(borders[1])
        y_max = np.max(borders[1]) + 1
        image = image[x_min:x_max, y_min:y_max]
        if mask is not None:
            mask = mask[x_min:x_max, y_min:y_max]
            return image, mask
        return image
    except:
        return image
class PandaDataset(Dataset):
    def __init__(self, images, folder, img_height, img_width, mode="train", rotate=0):
        self.images = images
        self.folder = folder
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width
        self.rotate = rotate
        
        # we are in validation part
        self.aug = albumentations.Compose([
            albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], always_apply=True)
        ])

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(self.folder, f'{img_name}.tiff')

        img = skimage.io.MultiImage(img_path)
        
        # TTA as needed
        if self.rotate==0:
            img = cv2.resize(img[-1], (1028, 1028))
 
        
        img = generate_tiles(img)
        img = cv2.resize(img, (self.img_height, self.img_width))
        
        img = Image.fromarray(img).convert("RGB")
        img = self.aug(image=np.array(img))["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        
        try: 
            os.remove(save_path)
        except: pass

        return torch.tensor(img, dtype=torch.float)
    
class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.b1 = nn.BatchNorm1d(in_f)
        self.d = nn.Dropout(0.75)
        
        self.l = nn.Linear(in_f, 512)
        self.r = nn.ReLU()
        self.b2 = nn.BatchNorm1d(512)
        
        self.o = nn.Linear(512, out_f)
        

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out
    
class FCN(torch.nn.Module):
    def __init__(self, base, in_f):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, config.CLASSES)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)
    
    def freeze_until(self, param_name):
        """
        Freeze layers of a model
        """
        found_name = False
        for name, params in self.named_parameters():
            if name == param_name:
                found_name = True
            params.requires_grad = found_name
# FOLD 1
model = get_model("efficientnet_b3b", pretrained=False)
model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
checkpoint = torch.load(model_path + "model_efficientnet_b3b_0_2.pth", map_location=config.device)    
model1 = FCN(model, 1536).to(config.device)  
model1.load_state_dict(checkpoint)
_ = model1.eval()
del checkpoint, model 

# FOLD 2
model = get_model("efficientnet_b3b", pretrained=False)
model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
checkpoint = torch.load(model_path + "model_efficientnet_b3b_1_2.pth", map_location=config.device)    
model2 = FCN(model, 1536).to(config.device)  
model2.load_state_dict(checkpoint)
_ = model2.eval()
del checkpoint, model 

# FOLD 3
model = get_model("efficientnet_b3b", pretrained=False)
model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
checkpoint = torch.load(model_path + "model_efficientnet_b3b_2_2.pth", map_location=config.device)    
model3 = FCN(model, 1536).to(config.device)  
model3.load_state_dict(checkpoint)
_ = model3.eval()
del checkpoint, model 

predictions = []

models = []
models.append(model1)
models.append(model2)
models.append(model3)

DEBUG = False
def check_for_images_dir():
    if DEBUG:
        return os.path.exists('../input/prostate-cancer-grade-assessment/train_images')
    else:
        return os.path.exists('../input/prostate-cancer-grade-assessment/test_images')

if DEBUG:
    test = train
    test_dir = data_dir

if check_for_images_dir():
    for model in models:
        model.eval()
        
        preds_all = []
        for rotate_id in range(1): # Maybe turn all rotation for final sub to beat overfit. Public LB was not improved so not doing TTA for now
            test_dataset = PandaDataset(
                test.image_id.values,
                test_dir,
                img_height=config.IMG_HEIGHT,
                img_width=config.IMG_WIDTH,
                mode="test",
                rotate=rotate_id
            )

            test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config.TEST_BATCH_SIZE,
                shuffle=False,
            )
            preds = []
            for idx, d in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
                inputs = d
                inputs = inputs.to(config.device)

                with torch.no_grad():
                    outputs = model(inputs)
                preds.append(outputs.to('cpu').numpy())
                #print(np.shape(preds))
            preds_all.append(np.concatenate(preds))

        predictions.append(np.mean(preds_all,axis=0))
                    
    predictions = np.mean(predictions, axis=0)
    predictions = predictions.argmax(1)


if len(predictions) > 0:
    submission.isup_grade = predictions
submission.isup_grade = submission['isup_grade'].astype(int)
submission.to_csv('submission.csv',index=False)
print(submission.head())   