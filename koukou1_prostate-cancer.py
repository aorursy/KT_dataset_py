# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import sys

sys.path = [

    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',

] + sys.path

import skimage.io

import numpy as np 

import pandas as pd 



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from efficientnet_pytorch import model as enet



import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
data_dir = '../input/prostate-cancer-grade-assessment'

df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

df_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))



model_dir = '../input/panda-enet-b1-model/'

image_folder = os.path.join(data_dir, 'test_images')

is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.

image_folder = image_folder if is_test else os.path.join(data_dir, 'train_images')



df = df_test if is_test else df_train.loc[:100]



tile_size = 256

image_size = 256

n_tiles = 36

batch_size = 8

num_workers = 4



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

        backbone = 'efficientnet-b1'

        model = enetv2(backbone, out_dim=5)

        model.load_state_dict(torch.load(model_f, map_location=lambda storage, loc: storage), strict=True)

        model.eval()

        model.to(device)

        models.append(model)

        print(f'{model_f} loaded!')

    return models







model_files = [

    'enet_8ep_fold0.pth'

]



models = load_models(model_files)
def get_tiles(img, mode=0):

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

        n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()

        if len(img) < n_tiles:

            img3 = np.pad(img3,[[0,N-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)

        idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]

        img3 = img3[idxs]

        for i in range(len(img3)):

            result.append({'img':img3[i], 'idx':i})

        return result, n_tiles_with_info >= n_tiles





class PANDADataset(Dataset):

    def __init__(self,

                 df,

                 image_size,

                 n_tiles=n_tiles,

                 tile_mode=0,

                 rand=False,

                 sub_imgs=False

                ):



        self.df = df.reset_index(drop=True)

        self.image_size = image_size

        self.n_tiles = n_tiles

        self.tile_mode = tile_mode

        self.rand = rand

        self.sub_imgs = sub_imgs



    def __len__(self):

        return self.df.shape[0]



    def __getitem__(self, index):

        row = self.df.iloc[index]

        img_id = row.image_id

        

        tiff_file = os.path.join(image_folder, f'{img_id}.tiff')

        image = skimage.io.MultiImage(tiff_file)[1]

        tiles, OK = get_tiles(image, self.tile_mode)



        if self.rand:

            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)

        else:

            idxes = list(range(self.n_tiles))

        idxes = np.asarray(idxes) + self.n_tiles if self.sub_imgs else idxes



        n_row_tiles = int(np.sqrt(self.n_tiles))

        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))

        for h in range(n_row_tiles):

            for w in range(n_row_tiles):

                i = h * n_row_tiles + w

    

                if len(tiles) > idxes[i]:

                    this_img = tiles[idxes[i]]['img']

                else:

                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255

                this_img = 255 - this_img

                h1 = h * image_size

                w1 = w * image_size

                images[h1:h1+image_size, w1:w1+image_size] = this_img



#         images = 255 - images

        images = images.astype(np.float32)

        images /= 255

        images = images.transpose(2, 0, 1)



        return torch.tensor(images)

dataset = PANDADataset(df, image_size, n_tiles,0)  # mode == 0

loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)



dataset2 = PANDADataset(df, image_size, n_tiles, 2)  # mode == 2

loader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers, shuffle=False)
LOGITS = []

LOGITS2 = []

with torch.no_grad():

    for data in tqdm(loader):

        data = data.to(device)

        logits = models[0](data)

        LOGITS.append(logits)



    for data in tqdm(loader2):

        data = data.to(device)

        logits = models[0](data)

        LOGITS2.append(logits)



LOGITS = (torch.cat(LOGITS).sigmoid().cpu() + torch.cat(LOGITS2).sigmoid().cpu()) / 2

PREDS = LOGITS.sum(1).round().numpy()



df['isup_grade'] = PREDS.astype(int)

df[['image_id', 'isup_grade']].to_csv('submission.csv', index=False)

print(df.head())

print()

print(df.isup_grade.value_counts())