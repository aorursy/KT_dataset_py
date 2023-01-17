import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
import torch

from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image

import torchvision

import torchvision.transforms as transforms

from torchvision.utils import make_grid

from torchvision.datasets import ImageFolder
from skmultilearn.model_selection import IterativeStratification
project_name = 'protein_classifier'
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'

TRAIN_DIR = DATA_DIR + '/train'

TEST_DIR = DATA_DIR + '/test'



TRAIN_CSV = DATA_DIR + '/train.csv'

TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv'
!head "{TRAIN_CSV}"
!head '{TEST_CSV}'
train_df = pd.read_csv(TRAIN_CSV)

train_df.head()
labels = {

    0: 'Mitochondria',

    1: 'Nuclear bodies',

    2: 'Nucleoli',

    3: 'Golgi apparatus',

    4: 'Nucleoplasm',

    5: 'Nucleoli fibrillar center',

    6: 'Cytosol',

    7: 'Plasma membrane',

    8: 'Centrosome',

    9: 'Nuclear speckles'

}
def encode_label(label):

    """Encodes the multi labels into a vector(tensor)."""

    target = torch.zeros(10)

    for l in str(label).split(' '):

        target[int(l)] = 1.

    return target



def decode_target(target, text_labels=False, threshold=0.5):

    """Decodes a tensor into a sequence of labels."""

    result = []

    for i, x in enumerate(target):

        if (x >= threshold):

            if text_labels:

                result.append(labels[i] + "(" + str(i) + ")")

            else:

                result.append(str(i))

    return ' '.join(result)
encode_label('2 4 5')
decode_target(torch.tensor([0., 0., 1., 0., 1., 1., 0., 0., 0., 0.]))
decode_target(torch.tensor([0, 0, 1, 0, 1, 1, 0, 0, 0, 0.]), text_labels=True)
class HumanProteinDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.df = pd.read_csv(csv_file)

        self.transform = transform

        self.root_dir = root_dir

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img_id, img_label = row['Image'], row['Label']

        img_fname = self.root_dir + "/" + str(img_id) + ".png"

        img = Image.open(img_fname)

        if self.transform:

            img = self.transform(img)

        return img, encode_label(img_label)
transform = transforms.Compose([transforms.ToTensor()])

dataset = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)
len(dataset)
def show_sample(img, target, invert=True):

    if invert:

        plt.imshow(1 - img.permute((1, 2, 0)))

    else:

        plt.imshow(img.permute(1, 2, 0))

    print('Labels:', decode_target(target, text_labels=True))
show_sample(*dataset[0], invert=False)
show_sample(*dataset[0], invert=True)
torch.manual_seed(10)
val_pct = 0.1

val_size = int(val_pct * len(dataset))

train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

len(train_ds), len(val_ds)
batch_size = 64
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

val_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)
def show_batch(dl, invert=True):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_xticks([]); ax.set_yticks([])

        data = 1-images if invert else images

        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))

        break
show_batch(train_dl, invert=False)
train_df.head()
train_df['Label'].value_counts().head()
train_df['Label'].value_counts().tail()
df = train_df.copy()
df = df.set_index('Image').sort_index()
df['Label'] = df['Label'].apply(lambda x: x.split(' '))
df.head()
df = df.explode('Label')

df.head()
df['Label'].value_counts()
df = pd.get_dummies(df);df
df = df.groupby(df.index).sum(); df.head()
df.columns = labels.keys() ; df.head()
X, y = df.index.values, df.values
k_fold = IterativeStratification(n_splits = 5, order=2)



splits = list(k_fold.split(X, y))
splits[0][0].shape , splits[0][1].shape
df.tail(), len(df)
splits[0][0], splits[0][1]
fold_splits = np.zeros(df.shape[0]).astype(int)



for i in range(5):

    fold_splits[splits[i][1]] = i # Note the validation fold set#



df['Split'] = fold_splits
df.tail(10)
train_df = df[df['Split'] != 0]

valid_df = df[df['Split'] == 0]
train_df.head()
valid_df.head()
from pathlib import Path

from tqdm.notebook import tqdm

import cv2



train_set = set(Path(TRAIN_DIR).iterdir())

test_set = set(Path(TEST_DIR).iterdir())

whole_set = train_set.union(test_set)



x_tot, x2_tot = [], []

for file in tqdm(whole_set):

   img = cv2.imread(str(file), cv2.COLOR_RGB2BGR)

   img = img/255.0

   x_tot.append(img.reshape(-1, 3).mean(0))

   x2_tot.append((img**2).reshape(-1, 3).mean(0))
#image stats

img_avr =  np.array(x_tot).mean(0)

img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)

print('mean:',img_avr, ', std:', np.sqrt(img_std))

mean = torch.as_tensor(x_tot)

std =torch.as_tensor(x2_tot)