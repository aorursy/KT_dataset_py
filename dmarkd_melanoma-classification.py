# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt

from typing import List
import os
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.nn import functional as F
from torchvision.utils import make_grid


from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import metrics

## custom dataset
# dataset
class ClassificationDataset:
    """classification dataset."""

    def __init__(self, images: List[str], cats, targets: List[str], transform=None):
        self.images = images
        self.cats = cats
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.images[idx])
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        cat = self.cats[idx, :]

        return image, cat, target
train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test_df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
## categorical features, embedding sizes
sns.distplot(train_df['age_approx'],kde=False)
pd.crosstab(train_df['target'],train_df['anatom_site_general_challenge']).apply(lambda r: r/r.sum(), axis=1)
pd.crosstab(train_df['target'],train_df['sex']).apply(lambda r: r/r.sum(), axis=1)
cat_vars = ["sex", "age_approx", "anatom_site_general_challenge"]
# Convert categorical columns to category dtypes.
for cat in cat_vars:
    train_df[cat] = train_df[cat].fillna("#na")
    train_df[cat] = train_df[cat].astype("category")
    test_df[cat] = test_df[cat].fillna("#na")
    test_df[cat] = test_df[cat].astype("category")
cat_szs = [len(train_df[col].cat.categories) for col in cat_vars]
emb_szs = [(size, min(50, (size + 1) // 2)) for size in cat_szs]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=0)

X = train_df[cat_vars].values
y = train_df.target.values

for trn_idx, vld_idx in split.split(X, y):
    train_index = trn_idx
    valid_index = vld_idx
new_train_df = train_df.loc[train_index]
valid_df = train_df.loc[valid_index]
train_targets = new_train_df["target"].values.astype(np.float32)
valid_targets = valid_df["target"].values.astype(np.float32)
test_targets = np.zeros(len(test_df)).astype(np.float32)
new_train_df['target'].value_counts()
valid_df['target'].value_counts()
train_images = [
    f"../input/melanoma/train_resized/train_resized/{img}.jpg" for img in new_train_df["image_name"].values
]
valid_images = [
    f"../input/melanoma/train_resized/train_resized/{img}.jpg" for img in valid_df["image_name"].values
]
test_images = [    f"../input/melanoma/test_resized/test_resized/{img}.jpg" for img in test_df["image_name"].values]
# # orig
# train_images = [
#     f"../input/siim-isic-melanoma-classification/jpeg/train/{img}.jpg" for img in new_train_df["image_name"].values
# ]
# valid_images = [
#     f"../input/siim-isic-melanoma-classification/jpeg/train/{img}.jpg" for img in valid_df["image_name"].values
# ]
# test_images = [
#     f"../input/siim-isic-melanoma-classification/jpeg/test/{img}.jpg" for img in test_df["image_name"].values
# ]
train_cats = torch.tensor(
    np.stack([new_train_df[col].cat.codes.values for col in cat_vars], 1), dtype=torch.int64
)
valid_cats = torch.tensor(
    np.stack([valid_df[col].cat.codes.values for col in cat_vars], 1), dtype=torch.int64
)
test_cats = torch.tensor(
    np.stack([test_df[col].cat.codes.values for col in cat_vars], 1), dtype=torch.int64
)
train_transform = transforms.Compose(
    [
        transforms.RandomRotation(30),      # rotate +/- 30 degrees
        transforms.RandomHorizontalFlip(),  # flip horizontal 50%
        transforms.RandomVerticalFlip(),  # flip vertical 50%
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),        
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.ToTensor(),        
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


train_data = ClassificationDataset(
    train_images, train_cats, train_targets, transform=train_transform
)
valid_data = ClassificationDataset(
    valid_images, valid_cats, valid_targets, transform=valid_transform
)
test_data = ClassificationDataset(
    test_images, test_cats, test_targets, transform=valid_transform
)
train_bs = 64
valid_bs = 911
test_bs = 38
np.random.seed(42)
torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size=train_bs, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_data, batch_size=valid_bs, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=test_bs, shuffle=False, pin_memory=True)
for images, cat, labels in train_loader:
    break

# Print the labels
print("Label:", labels.numpy())

im = make_grid(images, nrow=5)  # the default nrow is 8

# Inverse normalize the images
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)
im_inv = inv_normalize(im)

# Print the images
plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
## model definition
class TabularCNNModel(nn.Module):
    def __init__(self, emb_szs, layers, p=0.5):
        super().__init__()

        # tabular part
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(0.04)
        layerlist = []
        n_emb = np.array([nf for ni, nf in emb_szs]).sum()
        n_in = n_emb

        # layers is a list of number of neurons
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))            
            layerlist.append(nn.Dropout(0.001))
            n_in = i

        self.tabular_layers = nn.Sequential(*layerlist)

        # image part
        # freeze model
        self.cnn = models.resnet34(pretrained=True)
        
#         for param in self.cnn.parameters():
#             param.requires_grad = False

        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 200), 
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(200),
            nn.Dropout(0.001),
#             nn.Linear(1000, 200), 
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(200),
#             nn.Dropout(0.01),            
        )

        # combined layerlist
        self.classifier = nn.Sequential(
            nn.Linear(400, 200), 
            nn.ReLU(inplace=True),            
            nn.BatchNorm1d(200),
            nn.Dropout(0.01),
            nn.Linear(200, 1)
        )

    def forward(self, x_cat, images):

        # tabular
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))

        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x = self.tabular_layers(x)

        # cnn
        y = self.cnn(images)

        # combined
        xy = torch.cat([x, y], 1)
        xy = self.classifier(xy)

        return xy.squeeze()
model = TabularCNNModel(emb_szs=emb_szs, layers=[1000, 500, 200])
torch.cuda.is_available()
gpumodel = model.cuda()
gpumodel
## loss, optimizer, scheduler
criterion = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.AdamW(gpumodel.parameters(), lr=1e-4)

optimizer = torch.optim.AdamW([{'params': gpumodel.cnn.parameters(), 'lr':1e-4 },
                              {'params': gpumodel.classifier.parameters(), 'lr':2e-2 }], 
                              lr=1e-3
                             )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, threshold=0.001, mode="max"
)
## training
def save_checkpoint(state, is_best, filename='checkpoint_34_new.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")
import time
epochs = 25

train_losses = []
val_losses = []
val_aucs = []

best_auc = 0

for i in range(epochs):
    start_time = time.time()
    
    gpumodel.train()

    # Run the training batches
    for img_train, cat_train, y_train in train_loader:
        
        img_cuda = img_train.cuda()
        cat_cuda = cat_train.cuda()
        y_cuda = y_train.cuda()
        # predict
        y_pred = gpumodel(cat_cuda, img_cuda)
        train_loss = criterion(y_pred, y_cuda)

        # Update parameters
        optimizer.zero_grad()
        train_loss.backward() 
        optimizer.step()

    train_losses.append(train_loss)
        
    model.eval()
    val_preds = []
    with torch.no_grad():
        for  img_val, cat_val, y_val in valid_loader:
            imgv_cuda = img_val.cuda()
            catv_cuda = cat_val.cuda()
            yv_cuda = y_val.cuda()
            
            # Apply the model
            y_pred = gpumodel(catv_cuda, imgv_cuda)
            val_loss = criterion(y_pred, yv_cuda)
            val_preds.append(y_pred.cpu().numpy())
        
        val_losses.append(val_loss)            
        val_preds = np.vstack(val_preds).ravel()
        val_auc = metrics.roc_auc_score(valid_targets, val_preds)
        val_aucs.append(val_auc)
    
    print(f"""epoch: {i}/{epochs}: train loss: {train_loss}, 
          valid loss: {val_loss}, valid AUC {val_auc}""")

    scheduler.step(val_auc)
    
    if val_auc > best_auc:
        best_auc = val_auc
        is_best= True
    else:
        is_best= False
    
    state = {'epoch': i + 1,
            'state_dict': gpumodel.state_dict(),
            'optim_dict' : optimizer.state_dict(),
            'best_auc': best_auc}
    save_checkpoint(state, is_best = is_best) # path to folder
    
    print(f'epoch took {(time.time()-start_time)//60} minutes')



plt.plot(train_losses, label='train')
plt.plot(val_losses, label='valid')
state = torch.load('checkpoint_50.pth.tar', map_location='cuda:0')
state['best_auc']
state.keys()
gpumodel.load_state_dict(state['state_dict'])
test_preds = []
with torch.no_grad():
    for  img_tst, cat_tst, y_tst in test_loader:
        img_cuda = img_tst.cuda()
        cat_cuda = cat_tst.cuda()
        y_cuda = y_tst.cuda()

        # Apply the model
        y_pred = gpumodel(cat_cuda, img_cuda)
        test_preds.append(y_pred.cpu().numpy())
    
    test_preds = np.vstack(test_preds).ravel()
from scipy.special import expit
test_df['target'] = expit(test_preds)
sub_df = test_df[['image_name','target']]
sub_df.to_csv('sub_50_6_6.csv', index=False)
