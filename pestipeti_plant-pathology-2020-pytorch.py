import os

import numpy as np

import pandas as pd



import albumentations as A

import cv2



import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torch.optim as optim



from tqdm.notebook import tqdm

from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch import ToTensorV2



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold, StratifiedKFold



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import warnings  

warnings.filterwarnings('ignore')



__print__ = print



def print(string):

    os.system(f'echo \"{string}\"')

    __print__(string)
DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'



SEED = 42

N_FOLDS = 5

N_EPOCHS = 10

BATCH_SIZE = 64

SIZE = 512
class PlantDataset(Dataset):

    

    def __init__(self, df, transforms=None):

    

        self.df = df

        self.transforms=transforms

        

    def __len__(self):

        return self.df.shape[0]

    

    def __getitem__(self, idx):

        image_src = DIR_INPUT + '/images/' + self.df.loc[idx, 'image_id'] + '.jpg'

        # print(image_src)

        image = cv2.imread(image_src, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values

        labels = torch.from_numpy(labels.astype(np.int8))

        labels = labels.unsqueeze(-1)

        

        if self.transforms:

            transformed = self.transforms(image=image)

            image = transformed['image']



        return image, labels
class PlantModel(nn.Module):

    

    def __init__(self, num_classes=4):

        super().__init__()

        

        self.backbone = torchvision.models.resnet18(pretrained=True)

        

        in_features = self.backbone.fc.in_features



        self.logit = nn.Linear(in_features, num_classes)

        

    def forward(self, x):

        batch_size, C, H, W = x.shape

        

        x = self.backbone.conv1(x)

        x = self.backbone.bn1(x)

        x = self.backbone.relu(x)

        x = self.backbone.maxpool(x)



        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)

        x = self.backbone.layer3(x)

        x = self.backbone.layer4(x)

        

        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)

        x = F.dropout(x, 0.25, self.training)



        x = self.logit(x)



        return x

transforms_train = A.Compose([

    A.RandomResizedCrop(height=SIZE, width=SIZE, p=1.0),

    A.Flip(),

    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),



    # Pixels

    A.OneOf([

        A.IAAEmboss(p=1.0),

        A.IAASharpen(p=1.0),

        A.Blur(p=1.0),

    ], p=0.5),



    # Affine

    A.OneOf([

        A.ElasticTransform(p=1.0),

        A.IAAPiecewiseAffine(p=1.0)

    ], p=0.5),



    A.Normalize(p=1.0),

    ToTensorV2(p=1.0),

])



transforms_valid = A.Compose([

    A.Resize(height=SIZE, width=SIZE, p=1.0),

    A.Normalize(p=1.0),

    ToTensorV2(p=1.0),

])
submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')

submission_df.iloc[:, 1:] = 0



submission_df.head()
dataset_test = PlantDataset(df=submission_df, transforms=transforms_valid)

dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
train_df = pd.read_csv(DIR_INPUT + '/train.csv')



# For debugging.

# train_df = train_df.sample(n=100)

# train_df.reset_index(drop=True, inplace=True)



train_labels = train_df.iloc[:, 1:].values



# Need for the StratifiedKFold split

train_y = train_labels[:, 2] + train_labels[:, 3] * 2 + train_labels[:, 1] * 3



train_df.head()
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros((train_df.shape[0], 4))
# Download pretrained weights.

model = PlantModel(num_classes=4)
class DenseCrossEntropy(nn.Module):



    def __init__(self):

        super(DenseCrossEntropy, self).__init__()

        

        

    def forward(self, logits, labels):

        logits = logits.float()

        labels = labels.float()

        

        logprobs = F.log_softmax(logits, dim=-1)

        

        loss = -labels * logprobs

        loss = loss.sum(-1)



        return loss.mean()

    
def train_one_fold(i_fold, model, criterion, optimizer, dataloader_train, dataloader_valid):

    

    train_fold_results = []



    for epoch in range(N_EPOCHS):



        # print('  Epoch {}/{}'.format(epoch + 1, N_EPOCHS))

        # print('  ' + ('-' * 20))

        os.system(f'echo \"  Epoch {epoch}\"')



        model.train()

        tr_loss = 0



        for step, batch in enumerate(dataloader_train):



            images = batch[0]

            labels = batch[1]



            images = images.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.float)

            

            outputs = model(images)

            loss = criterion(outputs, labels.squeeze(-1))                

            loss.backward()



            tr_loss += loss.item()



            optimizer.step()

            optimizer.zero_grad()



        # Validate

        model.eval()

        val_loss = 0

        val_preds = None

        val_labels = None



        for step, batch in enumerate(dataloader_valid):



            images = batch[0]

            labels = batch[1]



            if val_labels is None:

                val_labels = labels.clone().squeeze(-1)

            else:

                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)



            images = images.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.float)



            with torch.no_grad():

                outputs = model(images)



                loss = criterion(outputs, labels.squeeze(-1))

                val_loss += loss.item()



                preds = torch.softmax(outputs, dim=1).data.cpu()



                if val_preds is None:

                    val_preds = preds

                else:

                    val_preds = torch.cat((val_preds, preds), dim=0)





        train_fold_results.append({

            'fold': i_fold,

            'epoch': epoch,

            'train_loss': tr_loss / len(dataloader_train),

            'valid_loss': val_loss / len(dataloader_valid),

            'valid_score': roc_auc_score(val_labels, val_preds, average='macro'),

        })



    return val_preds, train_fold_results
submissions = None

train_results = []



for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):

    print("Fold {}/{}".format(i_fold + 1, N_FOLDS))



    valid = train_df.iloc[valid_idx]

    valid.reset_index(drop=True, inplace=True)



    train = train_df.iloc[train_idx]

    train.reset_index(drop=True, inplace=True)    



    dataset_train = PlantDataset(df=train, transforms=transforms_train)

    dataset_valid = PlantDataset(df=valid, transforms=transforms_valid)



    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)



    device = torch.device("cuda:0")



    model = PlantModel(num_classes=4)

    model.to(device)



    criterion = DenseCrossEntropy()

    plist = [{'params': model.parameters(), 'lr': 5e-5}]

    optimizer = optim.Adam(plist, lr=5e-5)

    

    val_preds, train_fold_results = train_one_fold(i_fold, model, criterion, optimizer, dataloader_train, dataloader_valid)

    oof_preds[valid_idx, :] = val_preds.numpy()

    

    train_results = train_results + train_fold_results



    model.eval()

    test_preds = None



    for step, batch in enumerate(dataloader_test):



        images = batch[0]

        images = images.to(device, dtype=torch.float)



        with torch.no_grad():

            outputs = model(images)



            if test_preds is None:

                test_preds = outputs.data.cpu()

            else:

                test_preds = torch.cat((test_preds, outputs.data.cpu()), dim=0)

    

    

    # Save predictions per fold

    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_preds, dim=1)

    submission_df.to_csv('submission_fold_{}.csv'.format(i_fold), index=False)



    # logits avg

    if submissions is None:

        submissions = test_preds / N_FOLDS

    else:

        submissions += test_preds / N_FOLDS



print("5-Folds CV score: {:.4f}".format(roc_auc_score(train_labels, oof_preds, average='macro')))
train_results = pd.DataFrame(train_results)

train_results.head(10)
fig = make_subplots(rows=2, cols=1)



colors = [

    ('#d32f2f', '#ef5350'),

    ('#303f9f', '#5c6bc0'),

    ('#00796b', '#26a69a'),

    ('#fbc02d', '#ffeb3b'),

    ('#5d4037', '#8d6e63'),

]



for i in range(N_FOLDS):

    data = train_results[train_results['fold'] == i]



    fig.add_trace(go.Scatter(x=data['epoch'].values,

                             y=data['train_loss'].values,

                             mode='lines',

                             visible='legendonly' if i > 0 else True,

                             line=dict(color=colors[i][0], width=2),

                             name='Train loss - Fold #{}'.format(i)),

                 row=1, col=1)



    fig.add_trace(go.Scatter(x=data['epoch'],

                             y=data['valid_loss'].values,

                             mode='lines+markers',

                             visible='legendonly' if i > 0 else True,

                             line=dict(color=colors[i][1], width=2),

                             name='Valid loss - Fold #{}'.format(i)),

                 row=1, col=1)

    

    fig.add_trace(go.Scatter(x=data['epoch'].values,

                             y=data['valid_score'].values,

                             mode='lines+markers',

                             line=dict(color=colors[i][0], width=2),

                             name='Valid score - Fold #{}'.format(i),

                             showlegend=False),

                 row=2, col=1)



fig.update_layout({

  "annotations": [

    {

      "x": 0.225, 

      "y": 1.0, 

      "font": {"size": 16}, 

      "text": "Train / valid losses", 

      "xref": "paper", 

      "yref": "paper", 

      "xanchor": "center", 

      "yanchor": "bottom", 

      "showarrow": False

    }, 

    {

      "x": 0.775, 

      "y": 1.0, 

      "font": {"size": 16}, 

      "text": "Validation scores", 

      "xref": "paper", 

      "yref": "paper", 

      "xanchor": "center", 

      "yanchor": "bottom", 

      "showarrow": False

    }, 

  ]

})



fig.show()
submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(submissions, dim=1)

submission_df.to_csv('submission.csv', index=False)
submission_df