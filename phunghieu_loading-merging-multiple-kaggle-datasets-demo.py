INPUT_DIR = '/kaggle/input/'



TEST_SIZE = 0.3

RANDOM_STATE = 128



BATCH_SIZE = 8

NUM_WORKERS = 0
import torch

from torch.utils.data import Dataset, DataLoader

from albumentations import Normalize, Compose

import numpy as np

import pandas as pd

import cv2

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

import os

import glob
class RandomFaceDataset(Dataset):

    def __init__(self, img_dirs, labels, preprocess=None):

        '''

        Parameters:

            img_dirs: The directories that contain face images.

                Each directory coresponding to a video in the original training data.

            labels: Corresponding labels {'FAKE': 1, 'REAL', 0} of videos

            

        '''

        self.img_dirs = img_dirs

        self.labels = labels

        self.preprocess = preprocess



    def __len__(self):

        return len(self.img_dirs)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()



        img_dir = self.img_dirs[idx]

        label = self.labels[idx]

        face_paths = glob.glob(f'{img_dir}/*.png')



        sample = face_paths[np.random.choice(len(face_paths))]

        

        face = cv2.imread(sample, 1)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)



        if self.preprocess is not None:

            augmented = self.preprocess(image=face)

            face = augmented['image']

        

        return {'face': face, 'label': np.array([label], dtype=float)}
all_train_dirs = glob.glob(INPUT_DIR + 'deepfake-detection-faces-*')

for i, train_dir in enumerate(all_train_dirs):

    print('[{:02}]'.format(i), train_dir)
all_dataframes = []

for train_dir in all_train_dirs:

    df = pd.read_csv(os.path.join(train_dir, 'metadata.csv'))

    df['path'] = df['filename'].apply(lambda x: os.path.join(train_dir, x.split('.')[0]))

    all_dataframes.append(df)



train_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
train_df
# Remove videos that don't have any face

train_df = train_df[train_df['path'].map(lambda x: os.path.exists(x))]
train_df
train_df['label'].replace({'FAKE': 1, 'REAL': 0}, inplace=True)
train_df
label_count = train_df.groupby('label').count()['filename']

print(label_count)
X = train_df['path'].to_numpy()

y = train_df['label'].to_numpy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
preprocess = Compose([

    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1)

])



train_dataset = RandomFaceDataset(

    img_dirs=X_train,

    labels=y_train,

    preprocess=preprocess

)

val_dataset = RandomFaceDataset(

    img_dirs=X_val,

    labels=y_val,

    preprocess=preprocess

)



train_dataloader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS

)

val_dataloader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS

)
for batch in tqdm(train_dataloader):

    face_batch = batch['face']

    label_batch = batch['label']

    

    print(type(face_batch), face_batch.shape)

    print(type(label_batch), label_batch.shape)



    break
for batch in tqdm(val_dataloader):

    face_batch = batch['face']

    label_batch = batch['label']

    

    print(type(face_batch), face_batch.shape)

    print(type(label_batch), label_batch.shape)



    break