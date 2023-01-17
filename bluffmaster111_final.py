!pip install ../input/spamspip/spams-2.6.1
!pip install ../input/imagecodecs/imagecodecs-2020.5.30-cp37-cp37m-manylinux2010_x86_64.whl
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DEBUG = False
import os
import sys
sys.path = [
    '../input/efficientnet/EfficientNet-PyTorch-master/',
    '../input/staintoolszip/',
] + sys.path
import skimage.io
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import model as enet

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import cv2
import staintools

data_dir = '../input/prostate-cancer-grade-assessment'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
df_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

model_dir = '../input/4stains'
image_folder = os.path.join(data_dir, 'test_images')
is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.
image_folder = image_folder if is_test else os.path.join(data_dir, 'train_images')
train_folder = os.path.join(data_dir, 'train_images')

df = df_test if is_test else df_train.loc[:1]

tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 12
num_workers = 12

device = torch.device('cuda')

print(image_folder)
class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
    
    
def load_models(model_files):
    models = []
    for model_f in model_files:
        model_f = os.path.join(model_dir, model_f)
        backbone = 'efficientnet-b0'
        model = enetv2(backbone, out_dim=5)
        model.load_state_dict(torch.load(model_f, map_location=lambda storage, loc: storage), strict=False)
        model.eval()
        model.to(device)
        models.append(model)
        print(f'{model_f} loaded!')
    return models


model_files = [
    'b0-stain-bb9_best_fold0.pth','b0-stain-964_best_fold1.pth','b0-stain-917_best_fold2.pth',
    'b0-stain-c11_best_fold3.pth'
]

models = load_models(model_files)
def sampleMeanStd(img):


    img  = img.astype("float32")

    b_ch=np.mean(img[:,:,0])
    g_ch=np.mean(img[:,:,1])
    r_ch=np.mean(img[:,:,2])

    #Individual channel-wise mean subtraction
    img -= np.array((b_ch,g_ch,r_ch))

    b_ch=np.std(img[:,:,0])
    g_ch=np.std(img[:,:,1])
    r_ch=np.std(img[:,:,2])

    img /= np.array((b_ch,g_ch,r_ch))

    return img

def get_tiles(img, mode=0,forReference=False):
        result = []
        h, w, c = img.shape
        pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
        pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

        img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)
        img3 = img2.reshape(
            img2.shape[0] // tile_size,
            tile_size,
            img2.shape[1] // tile_size,
            tile_size,
            3
        )

        img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
        n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 245).sum()
#         print(n_tiles_with_info)
        if len(img) < n_tiles:
            img3 = np.pad(img3,[[0,N-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
        if forReference:
            idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]#[:]
        else:
            idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles_with_info]#[:n_tiles]
        img3 = img3[idxs]
        for i in range(len(img3)):
            result.append({'img':img3[i], 'idx':i})
        return result, n_tiles_with_info
    
def getItemCustom(img_id,n_tiles=36):
    tiff_file = os.path.join(train_folder, f'{img_id}.tiff')
    image = skimage.io.MultiImage(tiff_file)[1]
    #print(image.shape)
    tiles, OK = get_tiles(image,forReference=True)

    
    idxes = list(range(n_tiles))
#     idxes = np.asarray(idxes) + n_tiles if self.sub_imgs else idxes

    n_row_tiles = int(np.sqrt(n_tiles))
    images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w

            if len(tiles) > idxes[i]:
                this_img = tiles[idxes[i]]['img']
            else:
                this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
            #this_img = 255 - this_img
            h1 = h * image_size
            w1 = w * image_size
            images[h1:h1+image_size, w1:w1+image_size] = this_img

#         images = 255 - images
#         images = images.astype(np.float32)
#         images /= 255
    #images = sampleMeanStd(images)
    #print(images.shape)
    #images = images.transpose(2, 0, 1)
    #print(images.shape)

    return images #torch.tensor(images)



reference_stain = getItemCustom('001c62abd11fa4b57bf7a6c603a11bb9')
reference_stain = np.uint8(reference_stain)
reference_stain = cv2.resize(reference_stain,(1024,1024))
reference_stain = staintools.LuminosityStandardizer.standardize(reference_stain)
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(reference_stain)

reference_stain = getItemCustom('008069b542b0439ed69b194674051964')
reference_stain = np.uint8(reference_stain)
reference_stain = cv2.resize(reference_stain,(1024,1024))
reference_stain = staintools.LuminosityStandardizer.standardize(reference_stain)
normalizer2 = staintools.StainNormalizer(method='vahadane')
normalizer2.fit(reference_stain)

reference_stain = getItemCustom('0005f7aaab2800f6170c399693a96917')
reference_stain = np.uint8(reference_stain)
reference_stain = cv2.resize(reference_stain,(1024,1024))
reference_stain = staintools.LuminosityStandardizer.standardize(reference_stain)
normalizer3 = staintools.StainNormalizer(method='vahadane')
normalizer3.fit(reference_stain)

reference_stain = getItemCustom('999a911f00a8647b3603859bf62c8c11')
reference_stain = np.uint8(reference_stain)
reference_stain = cv2.resize(reference_stain,(1024,1024))
reference_stain = staintools.LuminosityStandardizer.standardize(reference_stain)
normalizer4 = staintools.StainNormalizer(method='vahadane')
normalizer4.fit(reference_stain)
class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 n_tiles=n_tiles,
                 tile_mode=0,
                 rand=False,
                 sub_imgs=False,
                 normalizer = None
                ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.sub_imgs = sub_imgs
        self.normalizer= normalizer

    def __len__(self):
        return self.df.shape[0]

#     def __getitem__(self, index):
#         row = self.df.iloc[index]
#         img_id = row.image_id
        
#         tiff_file = os.path.join(image_folder, f'{img_id}.tiff')
#         image = skimage.io.MultiImage(tiff_file)[1]
#         tiles, OK = get_tiles(image, self.tile_mode)

#         if self.rand:
#             idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
#         else:
#             idxes = list(range(self.n_tiles))
#         idxes = np.asarray(idxes) + self.n_tiles if self.sub_imgs else idxes

#         n_row_tiles = int(np.sqrt(self.n_tiles))
#         images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
#         for h in range(n_row_tiles):
#             for w in range(n_row_tiles):
#                 i = h * n_row_tiles + w
    
#                 if len(tiles) > idxes[i]:
#                     this_img = tiles[idxes[i]]['img']
#                 else:
#                     this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
#                 #this_img = 255 - this_img
#                 h1 = h * image_size
#                 w1 = w * image_size
#                 images[h1:h1+image_size, w1:w1+image_size] = this_img

#         image_to_transform = np.copy(images)
#         image_to_transform = np.uint8(image_to_transform)
#         image_to_transform = cv2.resize(image_to_transform,(1024,1024))
#         try:
#             image_to_transform = staintools.LuminosityStandardizer.standardize(image_to_transform)
#             normalized_image = self.normalizer.transform(image_to_transform)
#             normalized_image = cv2.resize(normalized_image,(1536,1536))
#             images = normalized_image
#         except Exception as e:
#             print(e)
#         images = 255 - images
#         images = images.astype(np.float32)
#         images /= 255
#         images = images.transpose(2, 0, 1)

#         return torch.tensor([images,images])
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        tiff_file = os.path.join(image_folder, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        tiles, tiles_count = get_tiles(image, self.tile_mode)
#         print(tiles_count)

#         if self.rand:
#             idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
#         else:
        idxes = list(range(tiles_count))
        if tiles_count < 36:
            idxes = list(range(self.n_tiles))
        if tiles_count > 45:
            idxes = list(range(self.n_tiles*2))
        #idxes = np.asarray(idxes) + self.n_tiles if self.sub_imgs else idxes
        
#         idxes1,tiles1 = idxes[:36],tiles[:36]
#         idxes2,tiles2 = idxes[36:],tiles[36:]

        n_row_tiles = int(np.sqrt(self.n_tiles))
        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
    
                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]['img']
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                #this_img = 255 - this_img
                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1+image_size, w1:w1+image_size] = this_img
        
        if tiles_count > 45:
            #create another image
            images2 = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
            for h in range(n_row_tiles):
                for w in range(n_row_tiles):
                    i = h * n_row_tiles + w

                    if len(tiles) > idxes[i+36]:
                        this_img = tiles[idxes[i+36]]['img']
                    else:
                        this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                    #this_img = 255 - this_img
                    h1 = h * image_size
                    w1 = w * image_size
                    images2[h1:h1+image_size, w1:w1+image_size] = this_img
        else:
            images2 = np.copy(images)

        image_to_transform = np.copy(images)
        image_to_transform = np.uint8(image_to_transform)
        image_to_transform = cv2.resize(image_to_transform,(1024,1024))
        try:
            image_to_transform = staintools.LuminosityStandardizer.standardize(image_to_transform)
            normalized_image = self.normalizer.transform(image_to_transform)
            normalized_image = cv2.resize(normalized_image,(1536,1536))
            images = normalized_image
        except Exception as e:
            print(e)
        images = 255 - images
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        
        if tiles_count > 45:
            image_to_transform = np.copy(images2)
            image_to_transform = np.uint8(image_to_transform)
            image_to_transform = cv2.resize(image_to_transform,(1024,1024))
            try:
                image_to_transform = staintools.LuminosityStandardizer.standardize(image_to_transform)
                normalized_image = self.normalizer.transform(image_to_transform)
                normalized_image = cv2.resize(normalized_image,(1536,1536))
                images2 = normalized_image
            except Exception as e:
                print(e)
            images2 = 255 - images2
            images2 = images2.astype(np.float32)
            images2 /= 255
            images2 = images2.transpose(2, 0, 1)
        else:
            images2 = 255 - images2
            images2 = images2.astype(np.float32)
            images2 /= 255
            images2 = images2.transpose(2, 0, 1)
            

        return torch.tensor([images,images2])#,['1','2']

# if not is_test:
#     dataset_show = PANDADataset(df, image_size, n_tiles, tile_mode=0,normalizer=normalizer)
#     from pylab import rcParams
#     rcParams['figure.figsize'] = 20,10
#     for i in range(2):
#         f, axarr = plt.subplots(1,5)
#         for p in range(5):
#             idx = np.random.randint(0, len(dataset_show))
#             img = dataset_show[idx]
#             axarr[p].imshow(1. - img.transpose(0, 1).transpose(1,2).squeeze())
#             axarr[p].set_title(str(idx))
dataset = PANDADataset(df, image_size, n_tiles, tile_mode=0,normalizer=normalizer)  # mode == 0
loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

dataset2 = PANDADataset(df, image_size, n_tiles, tile_mode=0,normalizer=normalizer2)  
loader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers, shuffle=False)

dataset3 = PANDADataset(df, image_size, n_tiles, tile_mode=0,normalizer=normalizer3)  
loader3 = DataLoader(dataset3, batch_size=batch_size, num_workers=num_workers, shuffle=False)

dataset4 = PANDADataset(df, image_size, n_tiles, tile_mode=0,normalizer=normalizer4)  
loader4 = DataLoader(dataset4, batch_size=batch_size, num_workers=num_workers, shuffle=False)
from time import time
# with torch.no_grad():
#     for data in tqdm(loader):
#         print(data.shape)
#         data = data.to(device)
#         logits = models[0](data)
#         print(logits.shape)
#         break
# logits[0],logits[1],logits[2],logits[3]
# torch.mean(logits,dim=0)
# LOGITS = []
# with torch.no_grad():
#     for data in tqdm(loader):
#         all_logits = []
#         for i in range(data.shape[0]):
#             img1,img2 = data[i][0],data[i][1]
#             logits = models[0](data[i].to(device))
#             logits = torch.mean(logits,dim=0)
#             all_logits.append(logits)

#         logits = torch.stack(all_logits)
#         LOGITS.append(logits)
        
                
start_time = time()
LOGITS = []
LOGITS2 = []
LOGITS3 = []
LOGITS4 = []
with torch.no_grad():
    for data in tqdm(loader):
        all_logits = []
        for i in range(data.shape[0]):
            img1,img2 = data[i][0],data[i][1]
            logits = models[0](data[i].to(device))
            logits = torch.mean(logits,dim=0)
            all_logits.append(logits)

        logits = torch.stack(all_logits)
        LOGITS.append(logits)

with torch.no_grad():
    for data in tqdm(loader2):
        all_logits = []
        for i in range(data.shape[0]):
            img1,img2 = data[i][0],data[i][1]
            logits = models[1](data[i].to(device))
            logits = torch.mean(logits,dim=0)
            all_logits.append(logits)

        logits = torch.stack(all_logits)
        LOGITS2.append(logits)

with torch.no_grad():
    for data in tqdm(loader3):
        all_logits = []
        for i in range(data.shape[0]):
            img1,img2 = data[i][0],data[i][1]
            logits = models[2](data[i].to(device))
            logits = torch.mean(logits,dim=0)
            all_logits.append(logits)

        logits = torch.stack(all_logits)
        LOGITS3.append(logits)

with torch.no_grad():
    for data in tqdm(loader4):
        all_logits = []
        for i in range(data.shape[0]):
            img1,img2 = data[i][0],data[i][1]
            logits = models[3](data[i].to(device))
            logits = torch.mean(logits,dim=0)
            all_logits.append(logits)

        logits = torch.stack(all_logits)
        LOGITS4.append(logits)
        
# with torch.no_grad():
#     for data in tqdm(loader4):
#         data = data.to(device)
#         logits = models[3](data)
#         LOGITS4.append(logits)

LOGITS_all = (torch.cat(LOGITS).sigmoid().cpu() + torch.cat(LOGITS2).sigmoid().cpu() + torch.cat(LOGITS3).sigmoid().cpu() + torch.cat(LOGITS4).sigmoid().cpu()) / 4
#LOGITS_all = (torch.cat(LOGITS).sigmoid().cpu() + torch.cat(LOGITS2).sigmoid().cpu() ) / 2
# LOGITS_all = torch.cat(LOGITS).sigmoid().cpu()
PREDS = LOGITS_all.sum(1).round().numpy()

df['isup_grade'] = PREDS.astype(int)
df[['image_id', 'isup_grade']].to_csv('submission.csv', index=False)
print('Total time taken',time()-start_time)
print(df.head())
print()
print(df.isup_grade.value_counts())

data.shape
