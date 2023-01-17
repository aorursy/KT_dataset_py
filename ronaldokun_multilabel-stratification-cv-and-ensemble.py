import os

import gc

import time

import copy

from pathlib import Path

import multiprocessing as mp

import random

import warnings

warnings.filterwarnings("ignore")



import cv2

import pandas as pd

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm







import torch

import torchvision.models as models

from torch.utils.data import Dataset, random_split, DataLoader

import torchvision.transforms as T

from sklearn.metrics import f1_score

import torch.nn.functional as F

import torch.nn as nn

from torchvision.utils import make_grid

from skmultilearn.model_selection import IterativeStratification

%matplotlib inline



      

ROOT = Path('/kaggle/input/jovian-pytorch-z2g/')

DIR = ROOT / 'Human protein atlas'

TRAIN = DIR / 'train'

TEST = DIR / 'test'

batch_size = 64

size = 256

nfolds = 5

threshold = 0.3

SEED = 2020
def show_sample(img, target, invert=True):

    if invert:

        plt.imshow(1 - img.permute((1, 2, 0)))

    else:

        plt.imshow(img.permute(1, 2, 0))

    print('Labels:', decode_target(target, text_labels=True))

    

def show_batch(dl, invert=True):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_xticks([]); ax.set_yticks([])

        data = 1-images if invert else images

        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))

        break



def F_score(output, label, threshold=0.5, beta=1):

    prob = output > threshold

    label = label > threshold



    TP = (prob & label).sum(1).float()

    TN = ((~prob) & (~label)).sum(1).float()

    FP = (prob & (~label)).sum(1).float()

    FN = ((~prob) & label).sum(1).float()



    precision = torch.mean(TP / (TP + FP + 1e-12))

    recall = torch.mean(TP / (TP + FN + 1e-12))

    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)

    return F2.mean(0)



def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')



def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)



class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        """Number of batches"""

        return len(self.dl)

    

class MultilabelImageClassificationBase(nn.Module):



    def training_step(self, batch):

        images, targets = batch 

        out = self(images)                      

        loss = F.binary_cross_entropy(out, targets)      

        return loss

    

    def validation_step(self, batch):

        images, targets = batch 

        out = self(images)                           # Generate predictions

        loss = F.binary_cross_entropy(out, targets)  # Calculate loss

        score = F_score(out, targets)

        return {'val_loss': loss.detach(), 'val_score': score.detach() }

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_scores = [x['val_score'] for x in outputs]

        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(

            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_score']))



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



def encode_label(label):

    target = torch.zeros(10)

    for l in str(label).split(' '):

        target[int(l)] = 1.

    return target



def decode_target(target, text_labels=False, threshold=0.5):

    result = []

    for i, x in enumerate(target):

        if (x >= threshold):

            if text_labels:

                result.append(labels[i] + "(" + str(i) + ")")

            else:

                result.append(str(i))

    return ' '.join(result)



seed_everything(SEED)
df = pd.read_csv(DIR / 'train.csv').set_index("Image").sort_index()

submission = pd.read_csv(ROOT / 'submission.csv') # Don't change the order in the submission

DEVICE = get_default_device()
train_images = {int(x.stem): x for x in TRAIN.iterdir() if x.suffix == '.png'}

test_images = {int(x.stem): x for x in TEST.iterdir() if x.suffix == '.png'}
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
def encode_label(label):

    target = torch.zeros(10)

    for l in str(label).split(' '):

        target[int(l)] = 1.

    return target



def decode_target(target, text_labels=False, threshold=0.5):

    result = []

    for i, x in enumerate(target):

        if (x >= threshold):

            if text_labels:

                result.append(labels[i] + "(" + str(i) + ")")

            else:

                result.append(str(i))

    return ' '.join(result)
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
indexes = {v:k for k,v in labels.items()}
df.Label.value_counts().tail(10)
df['Label'] = df.Label.str.split(" ") ; df.head()
df = df.explode('Label') ; df.head(10)
df.Label.value_counts()
df = pd.get_dummies(df) ; df.head()
df = df.groupby(df.index).sum() ; df.head()
df.columns = labels.keys() ; df.head()
X, y = df.index.values, df.values
k_fold = IterativeStratification(n_splits=nfolds, order=2)



splits = list(k_fold.split(X, y))
splits[0][0].shape , splits[0][1].shape
fold_splits = np.zeros(df.shape[0]).astype(np.int)



for i in range(nfolds):

    fold_splits[splits[i][1]] = i



df['Split'] = fold_splits



df.head(10)
fold = 0



train_df = df[df.Split != fold]

val_df = df[df.Split == fold]
train_df.head()
val_df.head()
decoded_train_df = pd.DataFrame({'Label' : list(map(decode_target, train_df.values))}, index=train_df.index)

decoded_val_df = pd.DataFrame({'Label' : list(map(decode_target, val_df.values))}, index=val_df.index)
decoded_train_df.Label.value_counts().tail(10)
decoded_val_df.Label.value_counts().tail(10)
def create_split_df(nfolds=5, order=1):



    df = pd.read_csv(DIR / 'train.csv').set_index("Image")



    submission = pd.read_csv(ROOT / 'submission.csv')



    split_df = pd.get_dummies(df.Label.str.split(" ").explode())



    split_df = split_df.groupby(split_df.index).sum() 



    X, y = split_df.index.values, split_df.values



    k_fold = IterativeStratification(n_splits=nfolds, order=order)



    splits = list(k_fold.split(X, y))



    fold_splits = np.zeros(df.shape[0]).astype(np.int)



    for i in range(nfolds):

        fold_splits[splits[i][1]] = i



    split_df['Split'] = fold_splits    



    df_folds = []



    for fold in range(nfolds):



        df_fold = split_df.copy()

            

        train_df = df_fold[df_fold.Split != fold].drop('Split', axis=1).reset_index()

        

        val_df = df_fold[df_fold.Split == fold].drop('Split', axis=1).reset_index()

        

        df_folds.append((train_df, val_df))



    return df_folds
splits = create_split_df(5, order=2)
#train_set = set(TRAIN.iterdir())

#test_set = set(TEST.iterdir())

#whole_set = train_set.union(test_set)



#x_tot, x2_tot = [], []

#for file in tqdm(whole_set):

#    img = cv2.imread(str(file), cv2.COLOR_RGB2BGR)

#    img = img/255.0

#    x_tot.append(img.reshape(-1, 3).mean(0))

#    x2_tot.append((img**2).reshape(-1, 3).mean(0))



#image stats

#img_avr =  np.array(x_tot).mean(0)

#img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)

#print('mean:',img_avr, ', std:', np.sqrt(img_std))

#mean = torch.as_tensor(x_tot)

#std =torch.as_tensor(x2_tot)
mean = torch.tensor([0.05438065, 0.05291743, 0.07920227])

std = torch.tensor([0.39414383, 0.33547948, 0.38544176])

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])

imagenet_std = torch.tensor([0.229, 0.224, 0.225])


train_tfms = T.Compose([

    T.Resize(size),

    T.RandomHorizontalFlip(), 

    T.RandomRotation(90), # Since the images are squares I experimented with 90º Rotation

    T.ToTensor(), 

    T.Normalize(mean, std, inplace=True), 

    T.RandomErasing(inplace=True)

])



valid_tfms = T.Compose([

    T.Resize(size), 

    T.ToTensor(), 

    T.Normalize(mean, std)

])
class HumanProteinDataset(Dataset):

    def __init__(self, df, transform=None, is_test=False):

        self.df = df

        self.transform = transform

        self.files = test_images if is_test else train_images

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img_id, img_label = int(row['Image']), row.drop('Image').values.astype(np.float32)

        img = self.files[img_id] 

        img = Image.open(img)

        if self.transform:

            img = self.transform(img)

        return img, img_label
class Proteinmodel(MultilabelImageClassificationBase):

    def __init__(self, encoder):

        super().__init__()

        # Use a pretrained model

        self.network = encoder(pretrained=True)

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
def get_split_dataloaders(split):

    train_df, val_df = split

    

    train_ds = HumanProteinDataset(train_df, transform=train_tfms)

    val_ds = HumanProteinDataset(val_df, transform=valid_tfms)

    

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=mp.cpu_count(), pin_memory=True)

    val_dl = DataLoader(val_ds, batch_size*2, num_workers=mp.cpu_count(), pin_memory=True)

    

    

    train_dl = DeviceDataLoader(train_dl, DEVICE)

    val_dl = DeviceDataLoader(val_dl, DEVICE)

    

    return train_dl, val_dl
def get_test_dl():

    test_ds = HumanProteinDataset(submission, transform=valid_tfms, is_test=True)

    test_dl = DataLoader(test_ds, batch_size*2, num_workers=mp.cpu_count(), pin_memory=True)

    return DeviceDataLoader(test_dl, DEVICE)
@torch.no_grad()

def evaluate(model, val_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in tqdm(val_loader)]

    return model.validation_epoch_end(outputs)



def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return param_group['lr']



def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 

                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, save_best='val_loss'):

    

    since = time.time()

    

    torch.cuda.empty_cache()

    history = []

    

    # Set up cutom optimizer with weight decay

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 

                                                steps_per_epoch=len(train_loader))

    

    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss, best_score = 1e4, 0.0

    

    for epoch in range(epochs):

        # Training Phase 

        model.train()

        train_losses = []

        lrs = []

        for batch in tqdm(train_loader):

            loss = model.training_step(batch)

            train_losses.append(loss)

            loss.backward()

            

            # Gradient clipping

            if grad_clip: 

                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            

            optimizer.step()

            optimizer.zero_grad()

            

            # Record & update learning rate

            lrs.append(get_lr(optimizer))

            sched.step()

        

        # Validation phase

        result = evaluate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        result['lrs'] = lrs

        model.epoch_end(epoch, result)

        

        if result['val_loss'] < best_loss:   

            best_loss = result['val_loss']

            if save_best == 'val_loss':

                best_model_wts = copy.deepcopy(model.state_dict())

        

            

        if result['val_score'] > best_score:

            best_score = result['val_score']                   

            if save_best == 'val_score':            

                best_model_wts = copy.deepcopy(model.state_dict())          

        

        history.append(result)

        

    time_elapsed = time.time() - since

    

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    

    print(f'Best val Score: {best_score:4f}')

    

    print(f'Best val loss: {best_loss:4f}')



    # load best model weights

    model.load_state_dict(best_model_wts)

        

    

    return model, history
def predict_single(image):

    xb = image.unsqueeze(0)

    xb = to_device(xb, device)

    preds = model(xb)

    prediction = preds[0]

    print("Prediction: ", prediction)

    show_sample(image, prediction)

    

@torch.no_grad()

def predict_dl(dl, model):

    torch.cuda.empty_cache()

    batch_probs = []

    for xb, _ in tqdm(dl):

        probs = model(xb)

        batch_probs.append(probs.cpu().detach())

    batch_probs = torch.cat(batch_probs)

    return batch_probs
device = get_default_device()
max_lr = 0.01

grad_clip = 0.1

weight_decay = 1e-4

opt_func = torch.optim.Adam





histories = []

predictions = []



test_dl = get_test_dl()



since = time.time()





for i, split in enumerate(splits):

    

    history = []

    

    train_dl, val_dl = get_split_dataloaders(split)

    

    # initialize parameters of model to train each fold from scratch and not leak info from different folds

    model = to_device(Proteinmodel(models.resnet50), device)

    

    model.freeze()    

    model, hist = fit_one_cycle(6, max_lr, model, train_dl, val_dl, 

                             grad_clip=grad_clip, 

                             weight_decay=weight_decay, 

                             opt_func=opt_func)

    

    history += hist

    

    model.unfreeze()   

    model, hist  = fit_one_cycle(4, max_lr/10, model, train_dl, val_dl, 

                             grad_clip=grad_clip, 

                             weight_decay=weight_decay, 

                             opt_func=opt_func)

    

    history += hist

    

    test_preds = predict_dl(test_dl, model)

    

    predictions.append(test_preds)

    

    del model

    

    gc.collect()

    

print(f'Total Training time: {(time.time() - since)/60:.2f} minutes')
prediction_cv = torch.stack(predictions).mean(axis=0)

decoded_predictions = test_preds > threshold

submission["Label"] = [decode_target(t.tolist()) for t in  decoded_predictions]

submission.to_csv("submission.csv", index=False)
submission.head(10)