import sys

sys.path = [

    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',

] + sys.path
from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.callbacks import *

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import cohen_kappa_score,confusion_matrix

import matplotlib.image as image

from tqdm.notebook import tqdm

import os

import gc

import zipfile

import openslide

import cv2

from PIL import Image

import skimage.io as sk

import warnings

# from torchsummary import summary

from sys import getsizeof

warnings.filterwarnings("ignore")
device = torch.device('cuda')
tile_size = 256

image_size = 256

n_tiles = 36

batch_size = 8

num_workers = 4

TRAIN = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'
sld = os.listdir(TRAIN)

sld = [x[:-5] for x in sld]
df_duplicates = pd.read_csv('../input/duplicates-panda/duplicates.csv')

duplicate_files = df_duplicates['file2'].tolist()

df = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')

df = df[df['image_id'].isin(sld)]

df = df[~df['image_id'].isin(duplicate_files)]

df.columns = ['fn', 'data_provider', 'isup_grade', 'gleason_score']
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

        return result
class TiffImageItemList(ImageList):

    def open(self,fn):

        path = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

        fl = path + str(fn)+'.tiff'

        img = sk.MultiImage(fl)[1]

        res = get_tiles(img)

        imgs = []

        for i in range(36):

            im = res[i%len(res)]['img']

            imgs.append(im)

        imgs = np.array(imgs)

        final_image = np.concatenate(np.array([np.concatenate(imgs[j:j+6],axis=1).astype(np.uint8) for j in range(0,36,6)]),axis=0)

        final_image = cv2.resize(final_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        

        return vision.Image(pil2tensor(final_image,np.float32).div_(255))
data = (TiffImageItemList.from_df(df,path='',cols='fn')

                          .split_by_rand_pct()

                          .label_from_df(cols='isup_grade')

                          .transform(get_transforms())

                          .databunch(num_workers=4,bs=batch_size)

                          .normalize(imagenet_stats))
data_img = data

len(data_img.train_ds), len(data_img.valid_ds), data_img.classes, data_img.train_ds[0][0].data.shape,data_img.c
kp = KappaScore()

kp.weights = 'quadratic'
from efficientnet_pytorch import model as enet

pretrained_model = {

    'efficientnet-b3': '../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth'

}



enet_type = 'efficientnet-b3'

out_dim = 6
class enetv2(nn.Module):

    def __init__(self, backbone, out_dim):

        super(enetv2, self).__init__()

        self.enet = enet.EfficientNet.from_name(backbone)

        self.enet.load_state_dict(torch.load(pretrained_model[backbone]))



        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)

        self.enet._fc = nn.Identity()



    def extract(self, x):

        return self.enet(x)



    def forward(self, x):

        x = self.extract(x)

        x = self.myfc(x)

        return x
arch = enetv2(enet_type, out_dim=out_dim)
learn = Learner(data_img, arch , metrics = [kp] , model_dir = '/kaggle/working/').to_fp16()
learn.load('../input/prostate-cancer-efnetb3-fastai-custom-datablock/best_model_ft');

learn = learn.to_fp32()
test_df = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/test.csv')

df.columns = ['image_id', 'data_provider', 'isup_grade', 'gleason_score']

data_dir = '../input/prostate-cancer-grade-assessment'

image_folder = os.path.join(data_dir, 'test_images')

is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.

image_folder = image_folder if is_test else os.path.join(data_dir, 'train_images')



test = test_df if is_test else df.sample(n=100)
def image_test(fn,image_folder):     

    path2 = image_folder +'/'

    fl = path2 + str(fn)+'.tiff'

    img = sk.MultiImage(fl)[1]

    res = get_tiles(img)

    imgs = []

    for i in range(36):

        im = res[i%len(res)]['img']

        imgs.append(im)

    imgs = np.array(imgs)

    final_image = np.concatenate(np.array([np.concatenate(imgs[j:j+6],axis=1).astype(np.uint8) for j in range(0,36,6)]),axis=0)

    final_image = cv2.resize(final_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

    return vision.Image(pil2tensor(final_image,np.float32).div_(255))
ts_name = test.image_id.values

pred = np.zeros(len(ts_name))

    

for j in tqdm(range(len(ts_name))):

    ans = int(learn.predict(image_test(ts_name[j],image_folder))[0])

    pred[j] = ans

        

out = pd.DataFrame({'image_id':ts_name,'isup_grade':pred.astype(int)})

out.to_csv('submission.csv',index=False)