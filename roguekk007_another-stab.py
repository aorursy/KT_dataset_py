%reload_ext autoreload

%autoreload 2

%matplotlib inline



import cv2

from tqdm import tqdm_notebook as tqdm

import fastai

from fastai.vision import *

import os

from mish_activation import *

import warnings

from albumentations import *

from albumentations.pytorch import ToTensor

warnings.filterwarnings("ignore")
!pip install /kaggle/input/needed-packages/efficientnet_pytorch-0.5.1/efficientnet_pytorch-0.5.1

!pip install /kaggle/input/needed-packages/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4
HEIGHT = 137

WIDTH = 236

SIZE = 224

bs = 128

arch = models.resnet18

MODEL = '/kaggle/input/grapheme-fast-ai-starter-using-resnet18/resnet18_model_fold_0.pth'

nworkers = 2



TEST = ['/kaggle/input/bengaliai-cv19/test_image_data_0.parquet',

        '/kaggle/input/bengaliai-cv19/test_image_data_1.parquet',

        '/kaggle/input/bengaliai-cv19/test_image_data_2.parquet',

        '/kaggle/input/bengaliai-cv19/test_image_data_3.parquet']



LABELS = '../input/bengaliai-cv19/train.csv'

model_name = 'resnet34'

sd_path = '/kaggle/input/bengalib0/2_13_r34.pth'



df = pd.read_csv(LABELS)

nunique = list(df.nunique())[1:-1]
from efficientnet_pytorch import EfficientNet

import pretrainedmodels

'''model portion'''

class to3channels(nn.Module):

    def __init__(self):

        super().__init__()

        pass 

    

    def forward(self, x):

        return torch.stack([x, x, x], dim=1)



class identity(nn.Module):

    def __init__(self):

        super().__init__()

    

    def forward(self, x):

        return x



class add_tail(nn.Module):

    def __init__(self, backbone, num_features):

        super().__init__()

        self.pre = to3channels()

        self.backbone = backbone

        self.fc1 = nn.Linear(num_features, 168)

        self.fc2 = nn.Linear(num_features, 11)

        self.fc3 = nn.Linear(num_features, 7)

    

    def forward(self, x):

        x = self.backbone(self.pre(x))

        return self.fc1(x), self.fc2(x), self.fc3(x)

    

# Define model from argument here

def get_model(model_name):

    if 'efficientnet' in model_name:

        backbone = EfficientNet.from_name(model_name, override_params={'num_classes': 1})

        num_features = backbone._fc.weight.shape[1]

        backbone._fc = identity()

    else:

        try:

            backbone = pretrainedmodels.__dict__[model_name](pretrained=None)

        except:

            print('Available models are:', pretrainedmodels.model_names)

            raise NotImplementedError

        num_features = backbone.last_linear.weight.shape[1]

        backbone.last_linear = identity()

    return add_tail(backbone, num_features)
model = get_model(model_name).cuda()

model.load_state_dict(torch.load(sd_path, map_location='cuda'));

model.eval();
HEIGHT = 137

WIDTH = 236

import numpy as np

def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(image, pad=8, cols=None, rows=None, force_apply=False):

    img0 = 255 - image

    dx, dy = 5, 5

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back

    xmin = xmin - dx if (xmin > dx) else 0

    ymin = ymin - dy if (ymin > dy) else 0

    xmax = xmax + dx if (xmax < WIDTH - dx) else WIDTH

    ymax = ymax + dy if (ymax < HEIGHT - dy) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly)+pad

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return {'image':img}
class GraphemeDataset(Dataset):

    def __init__(self, df, transform):

        self.df = df

        self.transform = transform

        self.data = self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        name = self.df.iloc[idx,0]

        #normalize each image by its max val

        img = self.data[idx].astype(np.uint8)

        img = self.transform(image=img)['image']

        return img, name



from functools import partial

tsfm = Compose([

                partial(crop_resize),

                Resize(SIZE, SIZE),

                ToTensor()

            ])
import gc

row_id,target = [],[]

for fname in TEST:

    df = pd.read_parquet(fname)

    ds = GraphemeDataset(df, tsfm)

    dl = DataLoader(ds, batch_size=bs, num_workers=nworkers, shuffle=False)

    with torch.no_grad():

        for x,y in tqdm(dl):

            x = x.cuda()

            p1,p2,p3 = model(x)

            p1 = p1.argmax(-1).view(-1).cpu()

            p2 = p2.argmax(-1).view(-1).cpu()

            p3 = p3.argmax(-1).view(-1).cpu()

            for idx,name in enumerate(y):

                row_id += [f'{name}_grapheme_root',f'{name}_vowel_diacritic',

                           f'{name}_consonant_diacritic']

                target += [p1[idx].item(),p2[idx].item(),p3[idx].item()]

    del df, ds, dl

    gc.collect()

                

sub_df = pd.DataFrame({'row_id': row_id, 'target': target})

sub_df.to_csv('submission.csv', index=False)

sub_df.head()