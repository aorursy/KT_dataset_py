!pip -q install pytorch-lightning --upgrade
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
%matplotlib inline
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as func

import pytorch_lightning as pl

from torchvision import transforms, datasets, models

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tqdm.notebook import trange, tqdm 

torch.manual_seed(13);
path = '/kaggle/input/super-ai-image-classification'
train_img_path = path + '/train/train/images'
test_img_path = path + '/val/val/images'
train_img_df = pd.read_csv('../input/super-ai-image-classification/train/train/train.csv')
test_files = list(os.listdir(test_img_path))        
test_img_df = pd.DataFrame(test_files, columns = ['id'])
train_img_df['dir'] =  f"{train_img_path}/" + train_img_df['id']
train_img_df.head()
test_img_df['dir'] =  f"{test_img_path}/" + test_img_df['id']
test_img_df['category'] = 0
test_img_df.head()
class ImageTransform():
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=mean)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=mean)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=mean)
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)
class ImageDataset(Dataset):
    def __init__(self, df, transform=None, phase='train'):
        self.img_df = df
        self.img_path = df['dir'].values
        self.labels = df['category'].values
        self.transform = transform
        self.phase = phase
        
    def __len__(self): 
        return len(self.img_df)
    
    def __getitem__(self,idx):
        img_path = self.img_path[idx]
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img, self.phase)
        label = self.labels[idx]
        return img_transformed, label
def get_fc_layers(fc_sizes, ps):
    fc_layers_list = []
    for fc_size,p in zip(fc_sizes, ps):
        fc_layers_list.append(nn.Linear(fc_size[0], fc_size[1]))
        fc_layers_list.append(nn.ReLU(inplace=True))
        fc_layers_list.append(nn.BatchNorm1d(fc_size[1]))
        fc_layers_list.append(nn.Dropout(p=p))
    return nn.Sequential(*fc_layers_list)
class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self, img_df, num_target_classes=10, criterion = nn.CrossEntropyLoss(), batch_size=32, test_size=0.1):
        super(ImagenetTransferLearning, self).__init__()
        self.img_train_df, self.img_valid_df = train_test_split(img_df,stratify=img_df['category'], test_size=test_size, random_state=13, shuffle=True)     
        self.train_dataset = ImageDataset(self.img_train_df, 
                                             ImageTransform(), 
                                             phase='train')
        self.val_dataset = ImageDataset(self.img_valid_df, 
                                           ImageTransform(), 
                                           phase='val')
        self.criterion = criterion
        self.num_target_classes = num_target_classes
        self.batch_size = batch_size

        self.model = models.resnet101(pretrained=True, progress=False)
        self.model.classifier = nn.Sequential(
            get_fc_layers(fc_sizes=[(2048,512),(512,128)],ps=[0.5,0.5]), 
            nn.Linear(128, self.num_target_classes))
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
    
        
    def forward(self, x):
        x = self.model(x)
        return x
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    def configure_optimizers(self):
        # [optimizer], [schedular]
        return [self.optimizer], []
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = self.criterion(preds, labels)
        _, preds = torch.max(preds, 1)
        correct = torch.sum(preds == labels).float() / preds.size(0)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_correct', correct, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = self.criterion(preds, labels)
        _, preds = torch.max(preds, 1)
        correct = torch.sum(preds == labels).float() / preds.size(0)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_correct', correct, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_correct = torch.stack([x['val_correct'] for x in outputs]).mean()
        torch.cuda.empty_cache()
        self.log('avg_val_loss', avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('avg_val_correct', avg_correct, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return avg_loss
criterion = nn.CrossEntropyLoss()
num_target_classes = 2
batch_size = 64
epoch = 8
test_size = 0.1
model = ImagenetTransferLearning(train_img_df, num_target_classes, criterion, batch_size, test_size)
checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath='model/', monitor='val_loss', mode='min', save_weights_only=True)
# earlystopping = pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=4)
trainer = pl.Trainer(
    max_epochs=epoch,
    gpus=[0],
    checkpoint_callback=checkpoint_callback, 
#     callbacks = [earlystopping]
)

trainer.fit(model)
PATH = "./model.ckpt"
model_load = ImagenetTransferLearning.load_from_checkpoint(PATH, img_df = train_img_df, num_target_classes=2)

for p1, p2 in zip(model.cuda().parameters(), model_load.cuda().parameters()):
    if p1.data.ne(p2.data).sum() > 0:
        print(False)
        break
print(True)
def prediction(test_img_df, model, batch_size = 128):
    test_img_ds = ImageDataset(test_img_df, ImageTransform(), phase='val')
    test_img_dl = DataLoader(test_img_ds, batch_size=batch_size, shuffle=False)

    id_list = test_img_df.id
    pred_list = []

    with torch.no_grad():
        for xb, yb in tqdm(test_img_dl):
            _ = model.cuda().eval()
            y_pred = model(xb.cuda())
            y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            pred_list.append(y_pred_tags.cpu().detach().numpy().astype(int))
            
    pred_list = np.concatenate(pred_list)
    result  = pd.DataFrame({
        'id': id_list,
        'category': pred_list
    })
    
    result.sort_values(by='id', inplace=True)
    result.to_csv('submission.csv', index=False)
    
    return result
batch_size = 128
res = prediction(test_img_df, model, batch_size)
