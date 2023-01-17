!pip -q install pytorch-lightning --upgrade
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
%matplotlib inline

import os
import random
import multiprocessing
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
from PIL import Image, ImageChops, ImageOps
from PIL.ImageFilter import (BLUR, SHARPEN)
import cv2

pl.seed_everything(13);
path = '/kaggle/input/thai-mnist-classification'
train_img_path = path + '/train'
test_img_path = path + '/test'
train_rules_df = pd.read_csv(path + '/train.rules.csv')
test_rules_df = pd.read_csv(path + '/test.rules.csv')
minst_map_df = pd.read_csv(path + '/mnist.train.map.csv')
train_df = train_rules_df.copy()
test_df = test_rules_df.copy()
train_img_df = minst_map_df.copy()
train_img_df['dir'] =  f"{train_img_path}/" + train_img_df['id']
train_img_df.head()
pattern_dict = {}
for F1 in range(-1, 10):
    for F2 in range(0, 10):
        for F3 in range(0,10):
            if F1 == -1: eq = int(F2 + F3)
            elif F1 == 0: eq = int(F2*F3)
            elif F1 == 1: eq = int(((F2-F3)**2)**0.5)
            elif F1 == 2: eq = int((F2+F3)*((F2-F3)**2)**0.5)
            elif F1 == 3: eq = int((((F3*(F3 +1) - F2*(F2-1))/2)**2)**0.5)
            elif F1 == 4: eq = int(50+(F2-F3))
            elif F1 == 5: eq = min(F2, F3)
            elif F1 == 6: eq = max(F2, F3)
            elif F1 == 7: eq = int(((F2*F3)%9)*11)
            elif F1 == 8: eq = (((F2**2)+1)*F2 + F3*(F3+1))%99
            elif F1 == 9: eq = 50 + F2
            pattern_dict[f"{str(F1)}_{str(F2)}_{str(F3)}"] = int(eq)
test_img = list(set(pd.concat([test_rules_df.feature1 ,test_rules_df.feature2,test_rules_df.feature3])))
test_img_df = pd.DataFrame()
test_img_df['id'] = test_img
test_img_df['category'] = -1
test_img_df['dir'] = test_img_path + "/" + test_img_df['id']
test_img_df = test_img_df.dropna()
test_img_df
class ImageTransform():
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=mean)
            ]),
            'val': transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=mean)
            ]),
            'test': transforms.Compose([
                transforms.Resize((128,128)),
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
        img = self.pre_process(img)
        img_transformed = self.transform(img, self.phase)
        label = self.labels[idx]
        return img_transformed, label
    
    
    def pre_process(self, img):
        bg = Image.new(img.mode, img.size, img.getpixel((0,0))) 
        diff = ImageChops.difference(img, bg) 
        bbox = diff.getbbox() 
        if bbox:
            img = img.crop(bbox)
        img = img.filter(BLUR)
        img = img.filter(BLUR)
        img = np.array(img)
        _,img = cv2.threshold(img, 254.999, 255, cv2.THRESH_BINARY)
        img = Image.fromarray(img)
        img = ImageOps.invert(img)
        return img
def get_fc_layers(fc_sizes, ps):
    fc_layers_list = []
    for fc_size,p in zip(fc_sizes, ps):
        fc_layers_list.append(nn.Linear(fc_size[0], fc_size[1]))
        fc_layers_list.append(nn.ReLU(inplace=True))
        fc_layers_list.append(nn.BatchNorm1d(fc_size[1]))
        fc_layers_list.append(nn.Dropout(p=p))
    return nn.Sequential(*fc_layers_list)
class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self, img_df, num_target_classes=10, criterion = nn.CrossEntropyLoss(), batch_size=64, test_size=0.1):
        super(ImagenetTransferLearning, self).__init__()
        self.img_train_df, self.img_valid_df = train_test_split(img_df,stratify=img_df['category'], test_size=test_size, random_state=13, shuffle=True)     
        self.train_dataset = ImageDataset(self.img_train_df, 
                                             ImageTransform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
                                             phase='train')
        self.val_dataset = ImageDataset(self.img_valid_df, 
                                           ImageTransform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
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
        self.log('train_acc', correct, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = self.criterion(preds, labels)
        _, preds = torch.max(preds, 1)
        correct = torch.sum(preds == labels).float() / preds.size(0)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', correct, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_correct = torch.stack([x['val_correct'] for x in outputs]).mean()
        torch.cuda.empty_cache()
        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('avg_val_acc', avg_correct, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return avg_loss
criterion = nn.CrossEntropyLoss()
num_target_classes = 10
batch_size = 32
epoch = 12
test_size = 0.15
model = ImagenetTransferLearning(train_img_df, num_target_classes, criterion, batch_size, test_size)
checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath='model/{epoch}-{val_acc:.4f}', monitor='val_acc', mode='max', save_weights_only=True, save_top_k=6)
# earlystopping = pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=4)
trainer = pl.Trainer(
    max_epochs=epoch,
    gpus=[0],
    checkpoint_callback=checkpoint_callback, 
    deterministic=True
#     callbacks = [earlystopping]
)

trainer.fit(model);
# model_checkpoint = os.listdir('./model')
# model_checkpoint
# test2num = {}
# batch_size = 128

# for m in model_checkpoint:
#     test2num_load = {}
    
#     PATH = f"./model/{m}"
#     model_load = ImagenetTransferLearning.load_from_checkpoint(PATH, img_df = train_img_df)
#     res_load = prediction(test_img_df, model_load, batch_size)
#     for row in res_load.iterrows():
#         f = row[1]['id']
#         num = row[1]['category']
#         test2num_load[f] = num
#     test2num[m] = test2num_load
# for img in img2num:
#     if len(set(img2num[img])) == 1:
#         img2num[img] = img2num[img][0]
# for img in img2num:
#     try:
#         if len(set(img2num[img])) != 1:
#             all_num = set(img2num[img])
#             for num in all_num:
#                 if img2num[img].count(num) == 3:
#                     img2num[img] = num
#                 elif img2num[img].count(num) == 2 and len(all_num) == 3:
#                     img2num[img] = num
#     except:
#         pass
# for img in img2num:
#     try:
#         if len(set(img2num[img])) != 1:
#             img2num[img] = random.choice(img2num[img])
#     except:
#         pass
def interpret(all_img_df, model, batch_size = 128):
    all_img_ds = ImageDataset(all_img_df, ImageTransform(), phase='val')
    all_img_dl = DataLoader(all_img_ds, batch_size=batch_size, shuffle=False)

    id_list = all_img_df.id
    true_list = all_img_df.category
    pred_list = []
    loss_list = []
    _criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in tqdm(all_img_dl):
            _ = model.cuda().eval()
            y_pred = model(xb.cuda())
            y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            pred_list.append(y_pred_tags.cpu().detach().numpy().astype(int))
            
    pred_list = np.concatenate(pred_list)
    
    return true_list, pred_list, loss_list
pool = multiprocessing.Pool(processes = 10)
results_of_processes = [pool.apply_async(os.system, args=(cmd, ), callback = None )
                        for cmd in [
                        f"tensorboard --logdir ./lightning_logs/ --host 0.0.0.0 --port 6006 &",
                        "./ngrok http 6006 &"
                        ]]
!curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
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
#     result.to_csv('submission.csv', index=False)
    
    return result
batch_size = 128
res = prediction(test_img_df, model, batch_size)
# Visualize Prediction
import random
id_list = []
fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')

for ax in axes.ravel():
    # Select Image
    i = random.choice(res['id'].values)
    label = res.loc[res['id'] == i, 'category'].values[0]
    img_path = f"{test_img_path}/{i}"
    img = Image.open(img_path)
    ax.set_title(label)
    ax.imshow(img)
test_df['num_feature1'] = test_df['feature1'].apply(lambda name : img2num.get(name, -1))
test_df['num_feature2'] = test_df['feature2'].apply(lambda name : img2num.get(name, -1))
test_df['num_feature3'] = test_df['feature3'].apply(lambda name : img2num.get(name, -1))
test_df["pattern"] = test_df['num_feature1'].astype(str)+"_"+ test_df['num_feature2'].astype(str) + "_" + test_df['num_feature3'].astype(str)
test_df['predict'] = test_df.pattern.apply(lambda x:int(pattern_dict.get(x)))
test_df.head()
test_df[["id", "predict"]].to_csv("submission_multi.csv", index=False)
