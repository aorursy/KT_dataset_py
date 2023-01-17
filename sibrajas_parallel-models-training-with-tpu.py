!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!export XLA_USE_BF16=1
import gc
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from sklearn import model_selection 
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm.notebook import tqdm

from sklearn.utils import multiclass
from sklearn.preprocessing import MultiLabelBinarizer

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

import time
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
torch.set_default_tensor_type('torch.FloatTensor')
ROOT_DIR = '../input/jovian-pytorch-z2g'
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'

TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images
TEST_DIR = DATA_DIR + '/test'                             # Contains test images

TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images
TEST_CSV = ROOT_DIR+'/submission.csv'   # Contains dummy labels for test image
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

public_threshold = 0.5
local_threshold = 0.5
def encode_label(label):
    #print(label)
    target = torch.zeros(10)
    #for l in str(label).split(' '):
    for l in label:
        target[int(l)] = 1.
    return target

def decode_target(target, text_labels=False, threshold=public_threshold):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)
class HumanProteinDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, img_label = row['Image'], row['Label']
        #print(img_label)
        img_fname = self.root_dir + "/" + str(img_id) + ".png"
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)
        return img, encode_label(img_label)

#mean = [0.0792, 0.0529, 0.0544]
#std = [0.1288, 0.0884, 0.1374]
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
train_transform = transforms.Compose([
            transforms.RandomCrop(512, padding=8, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)])
test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)])

! pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
df = pd.read_csv(TRAIN_CSV)
df['kfold'] = -1
df = df.sample(frac=1).reset_index(drop=True)
mskf = MultilabelStratifiedKFold(n_splits=5)
vals = df.Label.values
y=[]
for x in vals:
    y.append([int(i) for i in x.split()])
y = MultiLabelBinarizer().fit_transform(y)
for f, (t_, v_) in enumerate(mskf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

df.to_csv("train_folds.csv", index=False)
del df
def F_score(output, label, threshold=local_threshold, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    #return F2.mean(0)
    return F2.mean()

train_batch_size = 16
valid_batch_size = 32

class ProteinCnnModel2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(fold):
    torch.manual_seed(42)
    model = ProteinCnnModel2()
    #model.load_state_dict(torch.load(f"../input/zerotogan-folds/model_{fold}.pt"))
    MX = xmp.MpModelWrapper(model)
    def get_train_split(fold):
        df = pd.read_csv("train_folds.csv")
        train_df = df[df.kfold != fold].reset_index(drop=True)
        val_df = df[df.kfold == fold].reset_index(drop=True)
        train_df['Label'] = [[int(i) for i in s.split()] for s in train_df['Label']]
        val_df['Label'] = [[int(i) for i in s.split()] for s in val_df['Label']]
        train = HumanProteinDataset(train_df, TRAIN_DIR, transform=train_transform)
        valid = HumanProteinDataset(val_df, TRAIN_DIR, transform=test_transform)
        return train, valid
    train, valid = get_train_split(fold)
    gc.collect()
    
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=train_batch_size,
        num_workers=2,
        pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid,
        batch_size=valid_batch_size,
        num_workers=1,
        pin_memory=True)

    learning_rate = 0.001
    lrs=[]

    # Get loss function, optimizer, and model
    device = xm.xla_device(fold+1)
    model = MX.to(device)
    model.freeze()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                           learning_rate, 
                           epochs=NUM_EPOCH,
                           steps_per_epoch=len(train_loader))
    
    def train_loop_fn(loader):
        model.train()
        tk0 = tqdm(loader, total=len(loader), desc="Training")
        for x, (data, target) in enumerate(tk0):
            optimizer.zero_grad()
            data  = data.to(device, dtype = torch.float, non_blocking=True)
            target  = target.to(device, dtype = torch.float, non_blocking=True)
            output = model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            lrs.append(get_lr(optimizer))
            scheduler.step()
            tk0.set_postfix(loss=loss.item())
            gc.collect()

    def test_loop_fn(loader):
        with torch.no_grad():
            model.eval()
            outputs = []
            tk0 = tqdm(loader, total=len(loader), desc="Validating")
            for x, (data, target) in enumerate(tk0):
                data  = data.to(device, dtype = torch.float, non_blocking=True)
                target  = target.to(device, dtype = torch.float, non_blocking=True)
                output = model(data)
                loss = F.binary_cross_entropy(output, target)  # Calculate loss
                score = F_score(output, target)
                vstep = {'val_loss': loss.cpu().detach(), 'val_score': score.cpu().detach() }
                outputs.append(vstep)
                tk0.set_postfix(loss=loss.item(), score=score.item())
                gc.collect()
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean().item()   # Combine losses
            batch_scores = [x['val_score'] for x in outputs]
            epoch_score = torch.stack(batch_scores).mean().item()      # Combine accuracies
            model.train()
        return epoch_score

    # Train and eval loops
    scores = []
    bestscore = 0
    for epoch in range(1, NUM_EPOCH + 1):
        gc.collect()
        start = time.time()
        train_loop_fn(train_loader)
        gc.collect()
        valscore= test_loop_fn(valid_loader)
        scores.append(valscore)
        gc.collect()
        if valscore > bestscore:
            print(f"Saving model for fold{fold} {bestscore} --> {valscore} in epoch-{epoch}")
            xm.save(model.state_dict(), "./model_{}.pt".format(fold))
            bestscore= valscore
        print("Fold {} epoch {} f1score {:.5f} in {:.2f} sec\n"\
                        .format(fold, epoch, np.mean(scores), time.time() - start))        
        
        if epoch == 7: #unfreeze
            model.unfreeze()
        gc.collect()
    return scores
from joblib import Parallel, delayed
Parallel(n_jobs=5, backend="threading")(delayed(train_model)(i) for i in range(5))
