#BASIC
import numpy as np 
import pandas as pd 
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
import torch.utils.model_zoo as model_zoo

import os
import sys

sys.path = [
    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',
] + sys.path

# DATA visualization
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import PIL
from IPython.display import Image, display
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


from tqdm import tqdm_notebook as tqdm
import math
import cv2

import openslide
import skimage.io
import random
import albumentations

from PIL import Image

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



BASE_FOLDER = "../input/prostate-cancer-grade-assessment/"
!ls {BASE_FOLDER}
class config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    TEST_BATCH_SIZE = 16
    CLASSES = 6
    DEBUG = False
#     DEBUG = True
#поскольку тестовые картинки для данного соревнования формируются в debag режиме, для дипломной работы взяты 
# тренировочные все за исключением 500 последних изображений биопсии, для тестовых - последние 500 из этого же файла
train = pd.read_csv(BASE_FOLDER+"train.csv")[:-500]
test = train[-500:]
#test = pd.read_csv(BASE_FOLDER+"test.csv")
sub = pd.read_csv(BASE_FOLDER+"sample_submission.csv")
data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'
mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks'


train_labels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
if config.DEBUG:
    data_dir = f'{BASE_PATH}/train_images'
    test = pd.read_csv(f'{BASE_PATH}/train.csv').head(200)
train.head()
train.describe().transpose()
train.isnull().sum()
print("unique ids : ", len(train.image_id.unique()))
train['data_provider'].value_counts()
train['isup_grade'].value_counts()
train['gleason_score'].value_counts()
train[train['gleason_score']=='negative']
print(len(train[train['gleason_score']=='0+0']['isup_grade']))
print(len(train[train['gleason_score']=='negative']['isup_grade']))
train['gleason_score'] = train['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)
print(len(train[train['gleason_score']=='0+0']['isup_grade']))
print(len(train[train['gleason_score']=='negative']['isup_grade']))
print(train[(train['gleason_score']=='3+4') | (train['gleason_score']=='4+3')]['isup_grade'].unique())
print(train[(train['gleason_score']=='3+5') | (train['gleason_score']=='5+3')]['isup_grade'].unique())
print(train[(train['gleason_score']=='5+4') | (train['gleason_score']=='4+5')]['isup_grade'].unique())
print(train[train['gleason_score']=='3+4']['isup_grade'].unique())
print(train[train['gleason_score']=='4+3']['isup_grade'].unique())
train[(train['isup_grade'] == 2) & (train['gleason_score'] == '4+3')]
train.drop([7273],inplace=True)
train.value_counts().sum()
test.head(10)

train['image_id']
((train.groupby('isup_grade')['image_id'].count()).sort_values(ascending=False))
fig = plt.figure(figsize=(10,6))
ax = sns.countplot(x="isup_grade", hue="data_provider",palette=["#bcbddc", "#efedf5"], data=train)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
                height +3,
                '{:1.2f}%'.format(100*height/10616),
                ha="center")
((train.groupby('gleason_score')['image_id'].count()).sort_values(ascending=False))

fig = plt.figure(figsize=(10,6))
ax = sns.countplot(x="gleason_score", hue="data_provider",palette=["#bcbddc", "#efedf5"], data=train)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
                height +3,
                '{:1.2f}%'.format(100*height/10616),
                ha="center")
def image_info(image,lev, coordinate, max_size=(600,400)):
    slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff'))
    # Здесь мы вычисляем "интервал между пикселями": физический размер пикселя изображения.
    # OpenSlide дает разрешение в сантиметрах, поэтому мы конвертируем его в микроны
    f,ax =  plt.subplots(2 ,figsize=(6,16))
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    #Выведем на экран увеличенное изображение
    patch = slide.read_region(coordinate, lev, (256, 256)) 
    ax[0].imshow(patch) 
    ax[0].set_title('Увеличенный отрезок биопсии')
    
    
    ax[1].imshow(slide.get_thumbnail(size=max_size)) #не увеличенное изображение
    ax[1].set_title('Полная биопсия')
    
    
    print(f"File id: {slide}")
    print(f"Размеры: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Количество слоев: {slide.level_count}")
    print(f"Уменьшение разрешения изображения в слое: {slide.level_downsamples}")
    print(f"Размеры слоя: {slide.level_dimensions}\n\n")
    print(f"ISUP стадия рака: {train.loc[train['image_id']==image, 'isup_grade']}")
    print(f"Шкала Глисон: {train.loc[train['image_id']==image, 'gleason_score']}")
train[train['isup_grade']==0]
image_info('0005f7aaab2800f6170c399693a96917', 0, (7000,15000))
image_info('ffb16f062dfe8fd1161eb29ad1bd80ab',0,  (2800,17500))
# '0005f7aaab2800f6170c399693a96917', 1, (7000,15000)
train['gleason_score'].value_counts()
train[train['gleason_score']=='3+3']
image_info('003046e27c8ead3e3db155780dc5498e',0, (17000,5600))
# image_info('ffb16f062dfe8fd1161eb29ad1bd80ab',0,  (2800,17500))
# '0005f7aaab2800f6170c399693a96917', 0, (7000,15000)
image_info('ff3667b139539475d4de2aa1b2325c82',0, (15000,2500))
# image_info('003046e27c8ead3e3db155780dc5498e',0, (17000,5600))
# image_info('ffb16f062dfe8fd1161eb29ad1bd80ab',0,  (2800,17500))
# '0005f7aaab2800f6170c399693a96917', 0, (7000,15000)
train[train['gleason_score']=='3+4'].head(1)
image_info('00bbc1482301d16de3ff63238cfd0b34', 0, (3000,8000))
# image_info('ff3667b139539475d4de2aa1b2325c82',0, (15000,2500))
# image_info('003046e27c8ead3e3db155780dc5498e',0, (17000,5600))
# image_info('ffb16f062dfe8fd1161eb29ad1bd80ab',0,  (2800,17500))
# '0005f7aaab2800f6170c399693a96917', 0, (7000,15000)
train[train['gleason_score']=='4+3'].head(1)
image_info('0068d4c7529e34fd4c9da863ce01a161', 0, (4000,8500))
# image_info('00bbc1482301d16de3ff63238cfd0b34', 0, (3000,8000))
# image_info('ff3667b139539475d4de2aa1b2325c82',0, (15000,2500))
# image_info('003046e27c8ead3e3db155780dc5498e',0, (17000,5600))
# image_info('ffb16f062dfe8fd1161eb29ad1bd80ab',0,  (2800,17500))
# '0005f7aaab2800f6170c399693a96917', 0, (7000,15000)
train[train['gleason_score']=='4+4'].head(1)
image_info('0018ae58b01bdadc8e347995b69f99aa', 0, (1000,8000))
# image_info('0068d4c7529e34fd4c9da863ce01a161', 0, (4000,8500))
# image_info('00bbc1482301d16de3ff63238cfd0b34', 0, (3000,8000))
# image_info('ff3667b139539475d4de2aa1b2325c82',0, (15000,2500))
# image_info('003046e27c8ead3e3db155780dc5498e',0, (17000,5600))
# image_info('ffb16f062dfe8fd1161eb29ad1bd80ab',0,  (2800,17500))
# '0005f7aaab2800f6170c399693a96917', 0, (7000,15000)
train[train['gleason_score']=='5+5'].head(1)
image_info('0403dcc49b1420545299f692f7d8e270', 0, (20000,4900))
# image_info('0018ae58b01bdadc8e347995b69f99aa', 0, (1000,8000))
# image_info('0068d4c7529e34fd4c9da863ce01a161', 0, (4000,8500))
# image_info('00bbc1482301d16de3ff63238cfd0b34', 0, (3000,8000))
# image_info('ff3667b139539475d4de2aa1b2325c82',0, (15000,2500))
# image_info('003046e27c8ead3e3db155780dc5498e',0, (17000,5600))
# image_info('ffb16f062dfe8fd1161eb29ad1bd80ab',0,  (2800,17500))
# '0005f7aaab2800f6170c399693a96917', 0, (7000,15000)
def look_canser(images):
    '''
    Функция позволяет загружать изображения через Оpenslide попеременно, работая с ними и затем, закрывая каждое
    '''
    f, ax = plt.subplots(2,3, figsize=(18,22))
    for i, image in enumerate(images):
        slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff')) 
        # Здесь мы вычисляем "интервал между пикселями": физический размер пикселя в изображении,
        #OpenSlide дает разрешение в сантиметрах, поэтому мы преобразуем его в микроны.
        spacing = 1/(float(slide.properties['tiff.XResolution']) / 10000)
        patch = slide.read_region((1780, 1920), 0, (256, 256)) 
        ax[i//3, i%3].imshow(patch) #Выводим изображения на экран
        slide.close()       
        ax[i//3, i%3].axis('off')
        #подписываем свои изображения
        image_id = image
        data_provider = train.loc[train['image_id']==image, 'data_provider']
        isup_grade = train.loc[train['image_id']==image, 'isup_grade']
        gleason_score = train.loc[train['image_id']==image, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

    plt.show() 
images = [
    '0403dcc49b1420545299f692f7d8e270',
    '035b1edd3d1aeeffc77ce5d248a01a53',
    '0018ae58b01bdadc8e347995b69f99aa',
    '0076bcb66e46fb485f5ba432b9a1fe8a',
    '068b0e3be4c35ea983f77accf8351cc8',
    '0838c82917cd9af681df249264d2769c',
]

look_canser(images)
train_labels.head(1)
label_mask =  openslide.OpenSlide(os.path.join(mask_dir, f'{"0005f7aaab2800f6170c399693a96917"}_mask.tiff'))
display(label_mask.get_thumbnail(size=(1000,200)))
def mask_info(slides):    
    f, ax = plt.subplots(2,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{slide}_mask.tiff'))
        mask_find = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

        ax[i//3, i%3].imshow(np.asarray(mask_find)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = train.loc[train['image_id']==slide, 'data_provider']
        isup_grade = train.loc[train['image_id']==slide, 'isup_grade']
        gleason_score = train.loc[train['image_id']==slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
        f.tight_layout()
        
    plt.show()
mask_info(images) 
def mask_vizual(image,max_size=(600,400)):
    slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff'))
    mask =  openslide.OpenSlide(os.path.join(mask_dir, f'{image}_mask.tiff'))
    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    f,ax =  plt.subplots(1,2 ,figsize=(18,22))
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    img = slide.get_thumbnail(size=(600,400)) #IMAGE 
    
    mask_find = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    
    ax[0].imshow(img) 
    #ax[0].set_title('Image')
    
    
    ax[1].imshow(np.asarray(mask_find)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) #IMAGE MASKS
    #ax[1].set_title('Image_MASK')
    
    
    image_id = image
    data_provider = train.loc[train['image_id']==image, 'data_provider']
    isup_grade = train.loc[train['image_id']==image, 'isup_grade']
    gleason_score = train.loc[train['image_id']==image, 'gleason_score']
    ax[0].set_title(f"ID: {image_id}\nSource: {data_provider}\n ISUP: {isup_grade}\n Gleason: {gleason_score} \nIMAGE")
    ax[1].set_title(f"ID: {image_id}\nSource: {data_provider} \nISUP: {isup_grade} \nGleason: {gleason_score} \nIMAGE_MASK")
mask_vizual('0403dcc49b1420545299f692f7d8e270')
for image in images:
    mask_vizual(image)
def carnser_or_normal(image_mask):
    mask =  openslide.OpenSlide(os.path.join(mask_dir, f'{image_mask}_mask.tiff'))
    mask_level = mask.read_region((0,0),mask.level_count - 1,mask.level_dimensions[-1]) #Selecting the level
    mask_find = np.asarray(mask_level)[:,:,0] #SELECTING R from RGB
    mask_background = np.where(mask_find == 0, 1, 0).astype(np.uint8) # SELECTING BG
    mask_benign = np.where(mask_find == 1, 1, 0).astype(np.uint8) #SELECTING BENIGN LABELS
    
    if (train[train['image_id'] == image_mask]['data_provider'] == 'karolinska').empty == True:
    #train.loc[image_mask,'data_provider'] == 'karolinska':
        mask_cancerous = np.where(mask_find == 2, 1, 0).astype(np.uint8) #SELECTING CANCEROUS LABELS
    else:
#     elif train.loc[image_mask,'data_provider'] == 'radboud':
        mask_cancerous = np.where(mask_find == 5, 1, 0).astype(np.uint8) #SELECTING NON-CANCEROUS LABELS
    
    return mask_background,mask_benign,mask_cancerous
background,benign,cancerous = carnser_or_normal('0403dcc49b1420545299f692f7d8e270')
image2 =[ '0403dcc49b1420545299f692f7d8e270',
    '0068d4c7529e34fd4c9da863ce01a161']

for image in image2:
    background,benign,cancerous = carnser_or_normal(image)

    #if train.loc[image,'data_provider'] == 'karolinska'
    fig, ax = plt.subplots(1, 3, figsize=(18, 12))

    ax[0].imshow(background.astype(float), cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('background');


#     ax[0].set_title('background,'+'  '+'data_provider:'+train.loc[image]["data_provider"]);
    ax[1].imshow(benign.astype(float), cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('benign');
    ax[2].imshow(cancerous.astype(float), cmap=plt.cm.gray)
    ax[2].axis('off')
    ax[2].set_title('cancerous')
train.head()
def find_canser(images, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Наложить маску на биопсию для различения региона раковых клеток"""
    f, ax = plt.subplots(2,3, figsize=(18,22))
    
    
    for i, image_id in enumerate(images):
        slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image_id}.tiff'))
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))
        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
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
            # Отображение: {0: фон (не ткань) или неизвестно, 1: соединительная ткань, 2: здоровый эпителий, 
            # 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Отображение: {0: фон (не ткань) или неизвестно, 1: доброкачественная,
            # 2: рак}
            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)

        mask_data.putpalette(data=preview_palette.tolist())
        mask_rgb = mask_data.convert(mode='RGB')
        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
        overlayed_image.thumbnail(size=max_size, resample=0)

        
        ax[i//3, i%3].imshow(overlayed_image) 
        slide.close()
        mask.close()       
        ax[i//3, i%3].axis('off')
        image_id = image
        data_provider = train.loc[train['image_id']==image, 'data_provider']
        isup_grade = train.loc[train['image_id']==image, 'isup_grade']
        gleason_score = train.loc[train['image_id']==image, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} \nISUP: {isup_grade} \nGleason: {gleason_score}")
find_canser(images)
test.head()
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=42)
from efficientnet_pytorch import EfficientNet 

class EfficientNetB3(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetB3, self).__init__()
        if pretrained == True:
            self.model = EfficientNet.from_name('efficientnet-b3')
            self.model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth'))
        else:
            self.model = EfficientNet.from_pretrained(None)            

        in_features = self.model._fc.in_features
        self.l0 = nn.Linear(in_features, config.CLASSES)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        return l0
    
class EfficientNetB0(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetB0, self).__init__()
        if pretrained == True:            
            self.model = EfficientNet.from_name('efficientnet-b0')
            self.model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth'))
        else:
            self.model = EfficientNet.from_pretrained(None)            

        in_features = self.model._fc.in_features
        self.l0 = nn.Linear(in_features, config.CLASSES)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        return l0

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
    if len(img3) < n_tiles:
        img3 = np.pad(img3,[[0,n_tiles-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img':img3[i], 'idx':i})
    return result, n_tiles_with_info >= n_tiles

def write_image(path, img):
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
tile_size = 256
image_size = 256
n_tiles = 16
batch_size = 2
class PANDADataset(Dataset):
    def __init__(self,
            df,
            image_size,
            n_tiles=n_tiles,
            tile_mode=0,
            rand=False,
        ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        # we are in validation part
        self.aug = albumentations.Compose([
            albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], always_apply=True)
        ])

    def __len__(self):
        return self.df.shape[0]
    

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        tiff_file = os.path.join(data_dir, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        tiles, OK = get_tiles(image, self.tile_mode)

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
        else:
            idxes = list(range(self.n_tiles))

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

                
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        
        
        img = images
        img = np.transpose(img, (1, 2, 0)) # orig image has shape(3,1024, 1024), converting to (1024, 1024, 3)
        img = 1 - img
        img = cv2.resize(img, (768, 768))
        write_image(f'{img_id}.png', img)
        
        # loading image
        
        img = skimage.io.MultiImage(f'{img_id}.png')[-1]
#         img = cv2.resize(img[-1], (512, 512))

        img = Image.fromarray(img).convert("RGB")
        img = self.aug(image=np.array(img))["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        

        return { 'image': torch.tensor(img, dtype=torch.float) }

ENSEMBLES = [
    {
        'model_name': 'efficientnet-b0',
        'model_weight': '../input/panda-inference-ensemble-trying-various-models/efficientnetB0_0.pth',
        'ensemble_weight': 1 
    },
    {
        'model_name': 'se_resnext50_32x4d',
        'model_weight': '../input/panda-open-models/resnext50_0.pth',
        'ensemble_weight': 1 
    }
]
device = config.device
models = []
for ensemble in ENSEMBLES:
    model = EfficientNetB3(pretrained=True)
    model.load_state_dict(torch.load(ensemble['model_weight'], map_location=device))
    model.to(device)
    models.append(model)

def check_for_images_dir():
    if config.DEBUG:
        return os.path.exists('../input/prostate-cancer-grade-assessment/train_images')
    else:
        return os.path.exists('../input/prostate-cancer-grade-assessment/train_images')      
model.eval()
predictions = []

if check_for_images_dir():
    
    test_dataset = PANDADataset(
        test,
        image_size,
        n_tiles,
        0
    )

# разбиваем изображение биопсии на 16 частей
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
    )

    for model in models:
        preds = []
        for idx, d in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            inputs = d["image"]
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            preds.append(outputs.to('cpu').numpy())
                    
        predictions.append(np.concatenate(preds))
    predictions = np.mean(predictions, axis=0)
    predictions = predictions.argmax(1)

test['y_pred'] = predictions

print(accuracy_score(test['isup_grade'], test['y_pred']))

print (f1_score(test['isup_grade'], test['y_pred'], average='weighted'))
test.head()