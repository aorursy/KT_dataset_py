import sys

import glob

import os



import albumentations as A

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.init as init

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from tqdm.notebook import tqdm_notebook as tqdm 

from PIL import Image



sys.path.insert(0, '../input/pretrained-models/pretrained-models.pytorch-master/')

import pretrainedmodels

IMG_HEIGHT = 137

IMG_WEIGHT = 236

TEST_BATCH_SIZE = 32

DEVICE = 'cuda'



USE_RGB = False



ImageNetStat = {

    'mean': [0.485, 0.456, 0.406],

    'std': [0.229, 0.224, 0.225]

}



BengaliAIStat = {

    'mean': [0.06922848809290576],

    'std': [0.20515700083327537]

}



STAT = ImageNetStat if USE_RGB else BengaliAIStat

class SqueezeNet(nn.Module):

    def __init__(self, pretrained=True, use_rgb=True):

        super(SqueezeNet, self).__init__()



        if pretrained:

            self.model = pretrainedmodels.models.squeezenet1_1(pretrained='imagenet')

        else:

            self.model = pretrainedmodels.models.squeezenet1_1(pretrained=None)



        if not use_rgb:

            # modify first layer

            first_conv = nn.Conv2d(1, 64, kernel_size=3, stride=2)

            init.kaiming_uniform_(first_conv.weight)

            self.model.features[0] = first_conv



        self.l0 = nn.Linear(512, 168)

        self.l1 = nn.Linear(512, 11)

        self.l2 = nn.Linear(512, 7)



    def forward(self, x):

        N = x.shape[0]



        x = self.model.features(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(N, -1)

        s0 = self.l0(x)

        s1 = self.l1(x)

        s2 = self.l2(x)

        return s0, s1, s2

import gc



class BengaliDatasetTest(Dataset):

    def __init__(self, df, aug=None, use_rgb=USE_RGB):

        self.img_arr = df.iloc[:, 1:].values

        self.image_id = df.image_id.values



        del df

        gc.collect()

    

        self.img_arr = self.img_arr.reshape(-1, 137, 236)

        self.aug = aug

        self.use_rgb = use_rgb



    def __getitem__(self, index):

        """ Get a sample from the dataset

        """

        image = Image.fromarray(self.img_arr[index])

        image_id = self.image_id[index]

        

        if self.use_rgb:

            image = image.convert('RGB')



        image = np.array(image)



        if self.aug is not None:

            image = self.aug(image=image)['image']



        image = image.astype(np.float32)



        if self.use_rgb:

            image = image.transpose((2, 0, 1))

        else:

            image = image[np.newaxis, :]



        return {

            'image': torch.tensor(image, dtype=torch.float32),

            'image_id': image_id

        }



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return len(self.image_id)
def predict(loader, model, fold):

    model.eval()

    model.load_state_dict(torch.load(fold))



    scores = {'g': [], 'v': [], 'c': []}

    with torch.no_grad():

        for i, d in tqdm(enumerate(loader), total=len(loader)):

            image = d['image'].to(DEVICE)

            g, v, c = model(image)

            g = g.cpu().numpy()

            v = v.cpu().numpy()

            c = c.cpu().numpy()

            for i in range(len(d['image'])):

                scores['g'].append(g[i])

                scores['v'].append(v[i])

                scores['c'].append(c[i])

    return scores
predictions = []



test_files = glob.glob('../input/bengaliai-cv19/test_image_data_*.parquet')

for f in tqdm(test_files, total=len(test_files)):

    df = pd.read_parquet(f)

    

    dataset = BengaliDatasetTest(

        df=df,

        aug=A.Compose([

            A.Resize(IMG_HEIGHT, IMG_WEIGHT, always_apply=True),

            A.Normalize(**STAT)

        ]),

        use_rgb=USE_RGB

    )

    

    loader = DataLoader(

        dataset=dataset,

        batch_size=TEST_BATCH_SIZE,

        num_workers=4,

        pin_memory=True

    )

    

    model = SqueezeNet(pretrained=False, use_rgb=USE_RGB)

    model.to(DEVICE)



    folds = glob.glob('../input/bengaliaicv19-squeezenet-pretrained/pretrained_models/squeezenet_train_folds_*.h5')

    gvc_scores = [predict(loader, model, fold) for fold in folds]



    

    g = np.mean([p['g'] for p in gvc_scores], axis=0)

    v = np.mean([p['v'] for p in gvc_scores], axis=0)

    c = np.mean([p['c'] for p in gvc_scores], axis=0)

    image_id = dataset.image_id

    

    g_preds = np.argmax(g, axis=1)

    v_preds = np.argmax(v, axis=1)

    c_preds = np.argmax(c, axis=1)



    for j, img_id in enumerate(image_id):

        predictions.append((f'{img_id}_consonant_diacritic', (c_preds[j])))

        predictions.append((f'{img_id}_grapheme_root', (g_preds[j])))

        predictions.append((f'{img_id}_vowel_diacritic', (v_preds[j])))
sub = pd.DataFrame(predictions, columns=['row_id', 'target'])

sub.head(10)
sub.to_csv('submission.csv', index=False)