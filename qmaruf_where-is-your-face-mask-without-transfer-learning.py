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
import matplotlib.pyplot as plt

from glob import glob

from sklearn import model_selection

import torch

import albumentations as albu

from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.callbacks import EarlyStopping

from sklearn import metrics

%matplotlib inline
early_stop_callback = EarlyStopping(

   monitor='val_loss',

   min_delta=0.0001,

   patience=5,

   verbose=False,

   mode='min',

)
with_mask = glob('/kaggle/input/face-mask-dataset/data/with_mask/*.jpg')

without_mask = glob('/kaggle/input/face-mask-dataset/data/without_mask/*.jpg')
fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))

axs = np.array(axs).ravel()

for i in range(100):

    axs[i].imshow(plt.imread(with_mask[i]))

    axs[i].grid('off')

    axs[i].axis('off')

# plt.suptitle('with mask images')

plt.show()

fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))

axs = np.array(axs).ravel()

for i in range(100):

    axs[i].imshow(plt.imread(without_mask[i]))

    axs[i].grid('off')

    axs[i].axis('off')

# plt.suptitle('without mask images')

plt.show()
faces = with_mask + without_mask

labels = [1]*len(with_mask) + [0]*len(without_mask)

faces_train, faces_test, labels_train, labels_test = model_selection.train_test_split(faces, labels, test_size=0.2, shuffle=True, random_state=1)
from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torchvision import transforms

from tqdm import tqdm

from PIL import Image

import cv2



class MyCustomDataset(Dataset):

    

    def read_face(self, face_path):

        face_img = cv2.imread(face_path)

        if face_img.shape[2] > 3:

            face_img = face_img[:, :, :3]

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        return face_img

        

    def __init__(self, faces, labels, split):

        self.faces = [self.read_face(face_path) for face_path in tqdm(faces)]

        self.labels = labels

        self.split = split

        

        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.aug_train = albu.Compose({

            albu.Resize(128, 128),

            albu.VerticalFlip(p=0.5),

            albu.Rotate(limit=(-10, 10)),

            albu.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        })

        

        self.aug_test = albu.Compose({

            albu.Resize(128, 128),            

            albu.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        })

        

    def __getitem__(self, index):

        face = self.faces[index]    

        if self.split == 'train':

            face = self.aug_train(image=np.array(face))['image']

        else:

            face = self.aug_test(image=np.array(face))['image']

        face = self.transforms(face)

        label = np.float32(self.labels[index])

        return face, label



    def __len__(self):

        return len(self.faces)
train_dataset = MyCustomDataset(faces_train, labels_train, 'train')

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)



test_dataset = MyCustomDataset(faces_test, labels_test, 'test')

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
import pytorch_lightning as pl

from torchvision import models

import torch.nn as nn



config = {

    'device': 'cuda:0',

    'learning_rate': 0.0001,

    'max_epochs': 10

}



class LitNet(pl.LightningModule):



    def __init__(self, train_dl, val_dl, test_dl):

        super(LitNet, self).__init__()        

        self.model = models.vgg16(pretrained=False)

        self.model.classifier[6] = nn.Linear(4096, 1)        

        self.criterion = nn.BCEWithLogitsLoss()



        self.learning_rate = config['learning_rate']   

        self.train_dl = train_dl

        self.val_dl = val_dl

        self.test_dl = test_dl



    def forward(self, x):

        return self.model(x)



    def training_step(self, batch, batch_nb):

        x, y = batch

        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y.view(y_hat.size()))

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}



    def validation_step(self, batch, batch_nb):

        x, y = batch

        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y.view(y_hat.size()))

        return {'val_loss': loss}



    def validation_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}



    def test_step(self, batch, batch_nb):

        x, y = batch

        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y.view(y_hat.size()))

        return {'test_loss': loss}



    def test_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss}

        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}



    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}



#     @pl.data_loader

    def train_dataloader(self):

        return self.train_dl



#     @pl.data_loader

    def val_dataloader(self):

        return self.val_dl



#     @pl.data_loader

    def test_dataloader(self):

        return self.test_dl
model = LitNet(train_dataloader, test_dataloader, test_dataloader)
trainer = pl.Trainer(max_epochs=config['max_epochs'],

                        gpus=1, 

                        check_val_every_n_epoch=1,

                        auto_lr_find=False,

                        early_stop_callback=early_stop_callback)    

trainer.fit(model)

trainer.test()
model.freeze()

op_sigmoid = nn.Sigmoid()



y_true, y_pred = [], []

for data in test_dataloader:

    imgs, lbls = data

    imgs, lbls = imgs.to(config['device']), lbls.to(config['device'])

    preds = model(imgs)

    y_pred.append(preds.data.cpu().numpy())

    y_true.append(lbls.data.cpu().numpy())

    

y_true = np.concatenate(y_true)

y_pred = np.concatenate(y_pred)
precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)

fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axs = np.array(axs).ravel()



axs[0].plot(recall, precision)

axs[0].set_xlabel('Recall')

axs[0].set_ylabel('Precision')

axs[0].set_title('Precision Recall Curve')





axs[1].plot(fpr, tpr)

axs[1].set_xlabel('FPR')

axs[1].set_ylabel('TPR')

axs[1].set_title('ROC Curve')



for a in range(2):

    axs[a].grid(True)

    axs[a].set_ylim([0, 1])
auroc = metrics.roc_auc_score(y_true, y_pred)

average_precision = metrics.average_precision_score(y_true, y_pred)



print ('auroc %f'%auroc)

print ('average precision %f'%average_precision)
class UnNormalize(object):

    def __init__(self, mean, std):

        self.mean = mean

        self.std = std



    def __call__(self, tensor):

        """

        Args:

            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:

            Tensor: Normalized image.

        """

        for t, m, s in zip(tensor, self.mean, self.std):

            t.mul_(s).add_(m)

            # The normalize code -> t.sub_(m).div_(s)

        return tensor

    

    

unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
import torchvision

to_pil = torchvision.transforms.ToPILImage()



false_negatives = []

false_positives = []



for data in test_dataloader:

    imgs, lbls = data

    imgs, lbls = imgs.to(config['device']), lbls.to(config['device'])

    preds = op_sigmoid(model(imgs))

    bin_preds = (preds > 0.5).data.cpu().numpy().astype(np.int)    

    lbls = lbls.data.cpu().numpy()

    for img, pred, lbl, bin_pred in zip(imgs, preds, lbls, bin_preds):

        if bin_pred[0] != lbl:

            if lbl == 1:

                false_negatives.append((img, pred[0]))

            else:

                false_positives.append((img, pred[0]))
nfn, nfp = len(false_negatives), len(false_positives)
def draw_grid(nrows, ncols, images):

    images = images[:nrows*ncols]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

    axs = np.array(axs).ravel()



    for i, dt in enumerate(images):    

        img = unorm(dt[0])

        img = to_pil(img.data.cpu())

        img = np.asarray(img)

        axs[i].imshow(img)

        axs[i].axis('off')

        axs[i].set_title('Prediction %f'%dt[1])

    

    plt.show()
# false negatives

draw_grid(nfn//3, 3, false_negatives)
# false positives

draw_grid(nfp//3, 3, false_positives)