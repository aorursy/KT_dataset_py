import os

import cv2

import torch

import torch.nn as nn

import torchvision.models as models

import torch.nn.functional as F

import torch.utils.data as data

import numpy as np

import pandas as pd

from pathlib import Path

import torchaudio

from collections import OrderedDict
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class FCN(nn.Module):

    def __init__(self, num_classes=264, pretrained=False):

        super().__init__()

        self.head = models.vgg16_bn(pretrained=pretrained)

        self.head.classifier[6] = nn.Sequential(

            nn.Linear(4096, 1024, bias=True),

            nn.BatchNorm1d(1024),

            nn.Dropout(.25),

            nn.ReLU(),

            nn.Linear(1024, 512, bias=True),

            nn.BatchNorm1d(512),

            nn.Dropout(.5),

            nn.ReLU(),

            nn.Linear(512, 264, bias=True),

        )

    

    def freeze(self, n_top=0, freeze_head=True):

        """

        :param n_top: Number of layers to freeze off the top classification layer.

        :param freeze_head: If the feature extractor of the network should be frozen.

        """

        self.head.features.requires_grad = not freeze_head

        for head in range(n_top):

            self.head.classifier[head].requires_grad = False

    

    def forward(self, x):

        x = x.repeat(1, 3, 1, 1)

        x = self.head(x)

        return torch.sigmoid(x)
model = FCN()
os.listdir('/kaggle/input/birdsong-recognition/')
TEST = Path('/kaggle/input/birdsong-recognition/test_audio').exists()

ROOT = Path('/kaggle/input/birdsong-recognition/')

if TEST:

    TEST_DIR = Path('/kaggle/input/birdsong-recognition/test_audio')

    META_DIR = Path('/kaggle/input/birdsong-recognition/')

else:

    TEST_DIR = Path('/kaggle/input/birdcall-check/test_audio')

    META_DIR = Path('/kaggle/input/birdcall-check/')

meta = pd.read_csv(META_DIR / 'test.csv')
class TestDataset(data.Dataset):

    def __init__(self, meta: pd.DataFrame, test_dir: Path, device, img_size=224):

        self.root = test_dir

        self.meta = meta

        self.device = device

        self.img_size = img_size

        self.amptodb = torchaudio.transforms.AmplitudeToDB()

    

    def __len__(self):

        return len(self.meta)

    

    def __getitem__(self, idx):

        sample = self.meta.iloc[idx]

        y, sr = torchaudio.load(self.root / f'{sample.audio_id}.mp3')

        y = y[0].reshape(y.shape[1])

        y = y.to(self.device)

        if sample.site in ['site_1', 'site_2']:

            start, end = sr * (sample.seconds - 5), sr * (sample.seconds)

            y = y[int(start):int(end)]

        else:

            start, end = sr * (0), sr * (5)

            y = y[int(start):int(end)]

        spectrogram = torchaudio.transforms.MelSpectrogram(sr, n_fft=2048, hop_length=512, f_min=10, f_max=1600).to(self.device)

        y = spectrogram(y).detach().cpu()

        y = self.amptodb(y)

        y = y.reshape([1, y.shape[1], y.shape[0]])

        

        return sample.row_id, y
train_meta = pd.read_csv(Path(ROOT / 'train.csv'))

ebird_codes = {index: code for index, code in enumerate(train_meta.ebird_code.unique())}
ebird_codes
model = FCN()

checkpoints = torch.load('/kaggle/input/pretrained-models-cornell-birdcall-recognition/model1.pth', map_location=device)



new_state_dict = OrderedDict()

for k, v in checkpoints.items():

    if 'module' not in k:

        k = 'module.'+k

    else:

        k = k[7:]

    new_state_dict[k]=v



model.load_state_dict(new_state_dict)

model = model.to(device)

model.eval()
TEST_DATASET = TestDataset(meta, TEST_DIR, device=device)

THRESHOLD = 0.80
submission = []

for sample in range(len(TEST_DATASET)):

    row_id, image = TEST_DATASET[sample]

    image = image.reshape((1, *image.shape))

    image = image.to(device, dtype=torch.float)

    

    outputs = model(image)

    outputs = outputs.cpu().detach().numpy()

    

    # Create output string

    indicies = (outputs>=THRESHOLD).nonzero()[1]

    if len(indicies) == 0:

        s = 'nocall'

    else:

        s = ' '.join([ebird_codes[i] for i in indicies])

    submission.append([row_id, s])

sub = pd.DataFrame(submission, columns=['row_id', 'birds'])
sub.to_csv('submission.csv', index=False)