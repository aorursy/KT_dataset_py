import os

import cv2

import glob

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tqdm

import torch

import torch.nn.functional as F

import torch.nn as nn

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

import albumentations as A

from albumentations.pytorch import ToTensor

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from transformers import get_cosine_schedule_with_warmup
!ls ../input/siim-isic-melanoma-classification
data_dir = '../input/siim-isic-melanoma-classification'
train = pd.read_csv(f'{data_dir}/train.csv')

train.head()
train['target'].value_counts()
train_images = glob.glob(f'{data_dir}/jpeg/train/*')

len(train_images), train_images[0]
fig=plt.figure(figsize=(25, 25))

columns = 3

rows = 3

for i in range(1, columns*rows +1):

    name = train_images[i].split('/')[-1]

    img = cv2.imread(train_images[i])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    plt.title(name)

    plt.axis('off')

plt.show()
def plot_imgs(dataset_show):

    from pylab import rcParams

    rcParams['figure.figsize'] = 20,20

    for i in range(2):

        f, axarr = plt.subplots(1,3)

        for p in range(3):

            idx = np.random.randint(0, len(dataset_show))

            data = dataset_show[idx]

            npimg = data['image'].numpy()

            axarr[p].imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

            axarr[p].set_title(idx)
def generate_transforms(img_size):

    

    train_transform = A.Compose([

                                A.Resize(img_size, img_size),

                                A.HorizontalFlip(),

                                A.Normalize(),

                                ToTensor()

                                ])

    

    valid_transform = A.Compose([

                                A.Resize(img_size, img_size),

                                A.Normalize(),

                                ToTensor()

                                ])

    return train_transform, valid_transform
class SIIM_Dataset(Dataset):

    def __init__(self, df, data_dir, mode ='train', transform=None):



        self.df = df 

        self.image_ids = df['image_name'].tolist()

        self.data_dir = data_dir

        self.mode = mode

        self.transform = transform



    def __len__(self):

        return len(self.image_ids)



    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        image_dir = os.path.join(self.data_dir, image_id + '.jpg')

        image = cv2.imread(image_dir)  

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     

        

        if self.transform is not None:

            augmented = self.transform(image=image)

            image = augmented['image']

            

        data = {}

        data['image'] =image

        data['image_id'] = image_id

        

        if self.mode == 'test':    

            return data

        else:

            label = self.df.loc[self.df['image_name']==image_id, 'target'].values[0]

            data['label'] = torch.tensor(label)

            return data
img_size = 512

train_transform , valid_transform = generate_transforms(img_size)
img_dir = '../input/siim-isic-melanoma-classification/jpeg/train'
df_show = train.iloc[:1000]

dataset_show = SIIM_Dataset(df_show, img_dir, transform=train_transform)

# plotting transformed version of dataset

plot_imgs(dataset_show)
def get_dataloader(df, img_dir, mode, img_size = 512, batch_size=64):



    train_transform, valid_transform = generate_transforms(img_size)



    datasets = SIIM_Dataset(df, img_dir, mode, transform= train_transform if mode =='train' else valid_transform)

    

    is_train = mode =='train'

    dataloader = DataLoader(datasets,

                            shuffle=is_train,

                            batch_size=batch_size if is_train else 2*batch_size,

                            drop_last=is_train,

                            num_workers=4,

                            pin_memory=False)

    return dataloader
train_0 = train[train['target']==0]

train_1 = train[train['target']==1]

train_0.shape, train_1.shape
train_0_sample = train_0.sample(584)

train_0_sample.shape
df_balanced = pd.concat([train_0_sample, train_1])

df_balanced['target'].value_counts()
train_df, valid_df = train_test_split(df_balanced, test_size=0.1, stratify=df_balanced['target'].values, random_state=1234)

train_df.shape, valid_df.shape
!pip install -q efficientnet_pytorch

from efficientnet_pytorch import EfficientNet
class Net(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        # 1280 is the number of neurons in last layer. is diff for diff. architecture

        self.dense_output = nn.Linear(1280, num_classes)



    def forward(self, x):

        feat = self.model.extract_features(x)

        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)

        return self.dense_output(feat)
train_loader = get_dataloader(train_df, img_dir, mode='train', img_size = 256, batch_size=16)

valid_loader = get_dataloader(valid_df, img_dir, mode='valid', img_size = 256, batch_size=16)
model = Net(num_classes=1).cuda()
def update_avg(curr_avg, val, idx):

    return (curr_avg * idx + val) / (idx + 1)
def evaluate_single_epoch(model, dataloader, criterion):



    model.eval()

    curr_loss_avg = 0

    valid_preds, valid_targets = [], [] 

    

    tbar = tqdm.tqdm(dataloader,  total=len(dataloader))

    with torch.no_grad():

        for batch_idx, data in enumerate(tbar):

            images = data['image'].cuda()

            labels = data['label'].cuda()



            logits = model(images)

            probs =torch.sigmoid(logits)



            valid_preds.append(probs.detach().cpu().numpy())

            valid_targets.append(labels.detach().cpu().numpy())

            

            loss = criterion(logits.squeeze(), labels.float())

            

            curr_loss_avg = update_avg(curr_loss_avg, loss, batch_idx)        

            tbar.set_description('loss: {:.4}'.format(curr_loss_avg.item()))



        valid_preds = np.concatenate(valid_preds)

        valid_targets = np.concatenate(valid_targets)

        roc_metric =  roc_auc_score(valid_targets, valid_preds) 

        #roc = roc_auc_score(targs[:,i], preds[:,i])



            

    return curr_loss_avg.item(), roc_metric
def train_single_epoch(model, dataloader, criterion, optimizer, scheduler):

    

    model.train()

    curr_loss_avg = 0

    tbar = tqdm.tqdm(dataloader,  total=len(dataloader))

    for batch_idx, data in enumerate(tbar):

        images = data['image'].cuda()

        labels = data['label'].cuda()

        

        scheduler.step()

        logits = model(images)     

        loss = criterion(logits.squeeze(), labels.float())

    

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        

        curr_loss_avg = update_avg(curr_loss_avg, loss, batch_idx)        

        tbar.set_description('loss: %.5f, lr: %.6f' % (curr_loss_avg.item(), optimizer.param_groups[0]['lr']))

    return curr_loss_avg.item()
# hyperparameters for training

num_epochs = 30

lr = 0.001

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=5000)
best_val_score = 0.0 

early_stop = 0

for epoch in range(num_epochs):

        

    # train phase 

    train_loss = train_single_epoch(model, train_loader, criterion, optimizer, scheduler)

       

    # valid phase

    val_loss, roc_metric = evaluate_single_epoch(model, valid_loader, criterion)



    print(f'Epoch: {epoch+1} | Train_loss: {train_loss:.5f} | Val loss: {val_loss:.5f} | roc_metric : {roc_metric:.5f}')

    

    if roc_metric > best_val_score:

        early_stop = 0

        best_val_score = roc_metric

        # save best model

        torch.save(model.state_dict(), 'best_checkpoint.pth')

    else:

        early_stop += 1

        if early_stop == 7: # stopping condition for training

            break
