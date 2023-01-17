!pip install pytorch_lightning torch_optimizer category_encoders efficientnet-pytorch
%matplotlib inline

import os

import cv2

import glob

import random

import itertools

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold

import category_encoders as ce



import torch

from torch import nn

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import RandomSampler, SequentialSampler

from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as albu

from albumentations.pytorch import ToTensorV2



import pytorch_lightning as pl

from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.metrics.classification import AUROC

from torch_optimizer import RAdam

from efficientnet_pytorch import EfficientNet



import warnings

warnings.filterwarnings('ignore')
os.listdir('../input')
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True
DEBUG = False



SEED = 42

seed_everything(SEED)
def load_data(data_dir):

    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    

    # set fold

    cv = GroupKFold(n_splits=5)

    train['fold'] = -1

    for i, (trn_idx, val_idx) in enumerate(cv.split(train, train['target'], groups=train['patient_id'].tolist())):

        train.loc[val_idx, 'fold'] = i



    img_paths = {

        'train': glob.glob(os.path.join(data_dir, 'train', '*.jpg')),

        'test': glob.glob(os.path.join(data_dir, 'test', '*.jpg'))

    }

    

    return train, test, img_paths
train, test, img_paths = load_data('../input/jpeg-melanoma-384x384')
train.head()
test.head()
def preprocessing_meta(train, test):

    train = train[['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'target', 'fold']]

    test = test[['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge']]

    test.loc[:, 'target'] = 0

    test.loc[:, 'fold'] = 0



    # Preprocessing

    train['age_approx'] /= train['age_approx'].max()

    test['age_approx'] /= test['age_approx'].max()

    train['age_approx'].fillna(0, inplace=True)

    test['age_approx'].fillna(0, inplace=True)

    for c in ['sex', 'anatom_site_general_challenge']:

        train[c].fillna('Nodata', inplace=True)

        test[c].fillna('Nodata', inplace=True)

    encoder = ce.OneHotEncoder(cols=['sex', 'anatom_site_general_challenge'], handle_unknown='impute')

    train = encoder.fit_transform(train)

    test = encoder.transform(test)



    test.drop(['target', 'fold'], axis=1, inplace=True)



    return train, test
train, test = preprocessing_meta(train, test)

features_num = len([f for f in train.columns if f not in ['image_name', 'patient_id', 'target', 'fold']])
train.head()
class MelanomaDataset(Dataset):

    def __init__(self, df, img_paths, transform=None, phase='train'):

        self.df = df

        self.features = [f for f in self.df.columns if f not in ['image_name', 'patient_id', 'target', 'fold']]

        self.img_paths = img_paths

        self.transform = transform

        self.phase = phase



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        if 'image_id' in self.df.columns:

            img_name = row['image_id']

        else:

            img_name = row['image_name']



        meta = row[self.features]

        meta = torch.tensor(meta, dtype=torch.float32)



        img_path = [path for path in self.img_paths if img_name in path][0]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)



        if self.transform is not None:

            img = self.transform(img, self.phase)

        else:

            img = torch.from_numpy(img.transpose((2, 0, 1)))

            img = img / 255.



        if self.phase == 'test':

            return img, meta, img_name

        else:

            label = row['target']

            label = torch.tensor(label, dtype=torch.float)



        return img, meta, label
class ENet(nn.Module):

    def __init__(self, output_size=1, model_name='efficientnet-b0', meta_features_num=11):

        super(ENet, self).__init__()

        self.enet = EfficientNet.from_name(model_name=model_name)

        self.fc = nn.Sequential(

            nn.Linear(in_features=meta_features_num, out_features=500),

            nn.BatchNorm1d(500),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2)

        )

        self.classification = nn.Linear(1500, out_features=output_size)



    def forward(self, x, d):

        out1 = self.enet(x)

        out2 = self.fc(d)

        out = torch.cat((out1, out2), dim=1)



        out = self.classification(out)



        return out
class ImageTransform:

    def __init__(self, img_size=512, input_res=512, data_dir='./input'):

        self.data_dir = data_dir

        self.transform = {

            'train': albu.Compose([

                albu.ImageCompression(p=0.5),

                albu.Rotate(limit=80, p=1.0),

                albu.OneOf([

                    albu.OpticalDistortion(),

                    albu.GridDistortion(),

                ]),

                albu.RandomSizedCrop(min_max_height=(int(img_size * 0.7), input_res),

                                     height=img_size, width=img_size, p=1.0),

                albu.HorizontalFlip(p=0.5),

                albu.VerticalFlip(p=0.5),

                albu.GaussianBlur(p=0.3),

                albu.OneOf([

                    albu.RandomBrightnessContrast(),

                    albu.HueSaturationValue(),

                ]),

                albu.CoarseDropout(max_holes=8, max_height=img_size // 8, max_width=img_size // 8, fill_value=0, p=0.3),

                albu.Normalize(),

                ToTensorV2(),

            ], p=1.0),



            'val': albu.Compose([

                albu.CenterCrop(height=img_size, width=img_size, p=1.0),

                albu.Normalize(),

                ToTensorV2(),

            ], p=1.0),



            'test': albu.Compose([

                albu.ImageCompression(p=0.5),

                albu.RandomSizedCrop(min_max_height=(int(img_size * 0.9), input_res),

                                     height=img_size, width=img_size, p=1.0),

                albu.HorizontalFlip(p=0.5),

                albu.VerticalFlip(p=0.5),

                albu.Transpose(p=0.5),

                albu.CoarseDropout(max_holes=8, max_height=img_size // 8, max_width=img_size // 8, fill_value=0, p=0.3),

                albu.Normalize(),

                ToTensorV2(),

            ], p=1.0)

        }



    def __call__(self, img, phase='train'):

        augmented = self.transform[phase](image=img)



        return augmented['image']
# Setting  #######################

label_smoothing = 0.2

pos_weight = 3.1





class MelanomaSystem(pl.LightningModule):

    def __init__(self, net, cfg, img_paths, train_df, test_df, transform):

        super(MelanomaSystem, self).__init__()

        self.net = net

        self.cfg = cfg

        self.img_paths = img_paths

        self.train_df = train_df

        self.test_df = test_df

        self.transform = transform

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        self.best_loss = 1e+9

        self.best_auc = None

        self.best_weight = None

        self.auc_list = []

        self.loss_list = []



    def prepare_data(self):

        # Split Train, Validation

        fold = self.cfg['fold']

        train = self.train_df[self.train_df['fold'] != fold].reset_index(drop=True)

        val = self.train_df[self.train_df['fold'] == fold].reset_index(drop=True)



        self.train_dataset = MelanomaDataset(train, self.img_paths['train'], self.transform, phase='train')

        self.val_dataset = MelanomaDataset(val, self.img_paths['train'], self.transform, phase='val')

        self.test_dataset = MelanomaDataset(self.test_df, self.img_paths['test'], self.transform, phase='test')



    def train_dataloader(self):

        return DataLoader(self.train_dataset,

                          batch_size=self.cfg['batch_size'],

                          pin_memory=True,

                          sampler=RandomSampler(self.train_dataset), drop_last=True)



    def val_dataloader(self):

        return DataLoader(self.val_dataset,

                          batch_size=self.cfg['batch_size'],

                          pin_memory=True,

                          sampler=SequentialSampler(self.val_dataset), drop_last=False)



    def test_dataloader(self):

        return DataLoader(self.test_dataset,

                          batch_size=self.cfg['batch_size'],

                          pin_memory=False,

                          shuffle=False, drop_last=False)



    def configure_optimizers(self):

        self.optimizer = RAdam(self.parameters(), lr=self.cfg['lr'], weight_decay=2e-5)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg['epoch_num'], eta_min=0)

        return [self.optimizer], [self.scheduler]



    def forward(self, x, d):

        return self.net(x, d)



    def step(self, batch):

        inp, d, label = batch

        out = self.forward(inp, d)



        if label is not None:

            # Label Smoothing

            label_smo = label.float() * (1 - label_smoothing) + 0.5 * label_smoothing

            loss = self.criterion(out, label_smo.unsqueeze(1))

        else:

            loss = None



        return loss, label, torch.sigmoid(out)



    def training_step(self, batch, batch_idx):

        loss, label, logits = self.step(batch)

        logs = {'train/loss': loss.item()}



        return {'loss': loss, 'logits': logits, 'labels': label}



    def validation_step(self, batch, batch_idx):

        loss, label, logits = self.step(batch)

        val_logs = {'val/loss': loss.item()}



        return {'val_loss': loss, 'logits': logits.detach(), 'labels': label.detach()}



    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        LOGITS = torch.cat([x['logits'] for x in outputs])

        LABELS = torch.cat([x['labels'] for x in outputs])



        # Skip Sanity Check

        auc = AUROC()(pred=LOGITS, target=LABELS) if LABELS.float().mean() > 0 else 0.5

        logs = {'val/epoch_loss': avg_loss.item(), 'val/epoch_auc': auc}

        

        self.loss_list.append(avg_loss.item())

        self.auc_list.append(auc)



        return {'avg_val_loss': avg_loss}



    def test_step(self, batch, batch_idx):

        inp, d, img_name = batch

        out = self.forward(inp, d)

        logits = torch.sigmoid(out)



        return {'preds': logits, 'image_names': img_name}



    def test_epoch_end(self, outputs):

        PREDS = torch.cat([x['preds'] for x in outputs]).reshape((-1)).detach().cpu().numpy()

        # [tuple, tuple]

        IMG_NAMES = [x['image_names'] for x in outputs]

        # [list, list]

        IMG_NAMES = [list(x) for x in IMG_NAMES]

        IMG_NAMES = list(itertools.chain.from_iterable(IMG_NAMES))



        res = pd.DataFrame({

            'image_name': IMG_NAMES,

            'target': PREDS

        })



        try:

            res['target'] = res['target'].apply(lambda x: x.replace('[', '').replace(']', ''))

        except:

            pass

        

        N = len(glob.glob(f'submission_*.csv'))

        filename = f'submission_{N}.csv'

        res.to_csv(filename, index=False)

        

        return {'res': res}
# config

cfg = {

    'img_size': 256,

    'batch_size': 64,

    'epoch_num': 20,

    'lr': 5e-5,

    'fold': 0

}



if DEBUG:

    train = train.sample(100)

    test = test.sample(100)

    cfg['epoch_num'] = 2





net = ENet(model_name='efficientnet-b2', meta_features_num=features_num)

transform = ImageTransform(img_size=cfg['img_size'], input_res=384)



model = MelanomaSystem(net, cfg, img_paths, train, test, transform)



checkpoint_callback = ModelCheckpoint(

    filepath='.',

    save_top_k=1,

    verbose=True,

    monitor='avg_val_loss',

    mode='min'

)



trainer = Trainer(

    max_epochs=cfg['epoch_num'],

    checkpoint_callback=checkpoint_callback,

    gpus=[0]

    )
# Train

trainer.fit(model)
fig,axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 6))

axes[0].plot(model.loss_list, color='b')

axes[0].set_title('Val Loss')

axes[0].set_xlabel('Epoch')



axes[1].plot(model.auc_list, color='r')

axes[1].set_title('Val roc_auc')

axes[1].set_xlabel('Epoch')



plt.tight_layout()

plt.show()
# Predict

TTA_num = 3 if DEBUG else 20



# Test

for i in range(TTA_num):

    trainer.test(model)
def summarize_submit(sub_list, filename='submission.csv'):

    res = pd.DataFrame()

    for i, path in enumerate(sub_list):

        sub = pd.read_csv(path)



        if i == 0:

            res['image_name'] = sub['image_name']

            res['target'] = sub['target']

        else:

            res['target'] += sub['target']

        os.remove(path)



    # min-max norm

    res['target'] -= res['target'].min()

    res['target'] /= res['target'].max()



    return res
sub_list = glob.glob(f'submission_*.csv')



res = summarize_submit(sub_list)
res.head()
fig = plt.figure(figsize=(16, 4))

plt.hist(res['target'], bins=20)

plt.title('Histgram of Predict')

plt.show()
res.to_csv('submission.csv', index=False)