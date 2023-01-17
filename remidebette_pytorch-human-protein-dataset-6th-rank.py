import copy
import time
import gc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from skmultilearn.model_selection import IterativeStratification

from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from tqdm import tqdm

from torchvision import models
from torchvision import transforms as T

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()



def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def decode_target(target, text_labels=False, threshold=0.5):
    """Converts a probabilities vector to string of the definitive results, based on the threshold value"""
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)



def show_sample(img, target, invert=True):
    """Display a sample in the notebook.
    
    In a python IDE, you should additionally call plt.show()"""
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', decode_target(target, text_labels=True))


def show_batch(dl, invert=True):
    """Displays a batch of samples in the notebook.
    
    In a python IDE, you should additionally call plt.show()"""
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        data = 1 - images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        break



def plot_scores(history):
    """Displays the evolution of the scores.
    
    In a python IDE, you should additionally call plt.show()"""
    scores = [x['val_score'] for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('F1 score vs. No. of epochs')


def plot_losses(history):
    """Displays the evolution of the losses.
    
    In a python IDE, you should additionally call plt.show()"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


def plot_lrs(history):
    """Displays the evolution of the learning rates.
    
    In a python IDE, you should additionally call plt.show()"""
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')


@torch.no_grad()
def predict_dl_one_hot(dl, model, threshold=0.5):
    """Evaluates the model on a pytorch DataLoader"""
    torch.cuda.empty_cache()
    batch_probs = []
    for xb, _ in tqdm(dl):
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    return batch_probs
        

def get_lr(optimizer):
    """Get the current learning rate from a pytorch optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

# The following gave wrong means due to the substitution of channels before the dataloader
# I do not know why the sd were also different from Roberto's, I leave it here for now.

#def online_mean_and_sd(loader):
#    """Compute the mean and sd of a Pytorch Dataloader
#
#        Var[x] = E[X^2] - E^2[X]
#
#        Caution: slow and CPU-heavy
#    """
#    cnt = 0
#    fst_moment = torch.empty(3)
#    snd_moment = torch.empty(3)
#
#    for data, _ in loader:
#
#        b, c, h, w = data.shape
#        nb_pixels = b * h * w
#        sum_ = torch.sum(data, dim=[0, 2, 3])
#        sum_of_square = torch.sum(data**2, dim=[0, 2, 3])
#        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
#        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
#
#        cnt += nb_pixels
#
#    return fst_moment, torch.sqrt(snd_moment - fst_moment**2)


def statistics_from_pictures(train, test):
    """Compute the mean and sd of the images

        Var[x] = E[X^2] - E^2[X]

        Caution: slow and CPU-heavy
    """
    train_set = set(Path(train).iterdir())
    #test_set = set(Path(test).iterdir())
    whole_set = train_set #.union(test_set)

    x_tot, x2_tot = [], []
    for file in tqdm(whole_set):
       img = cv2.imread(str(file), cv2.COLOR_RGB2BGR)
       img = img/255.0
       x_tot.append(img.reshape(-1, 3).mean(0))
       x2_tot.append((img**2).reshape(-1, 3).mean(0))

    #image stats
    img_avr =  np.array(x_tot).mean(0)
    img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
    print('mean:',img_avr, ', std:', np.sqrt(img_std))
    # mean = torch.as_tensor(x_tot)
    # std =torch.as_tensor(x2_tot)

    return img_avr, img_std


def F_score(output, label, threshold=0.5, beta=1):
    """Provides a usable score system for evaulating the models' performances"""
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
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'
TRAIN_DIR = DATA_DIR + '/train'
TEST_DIR = DATA_DIR + '/test'
TRAIN_CSV = DATA_DIR + '/train.csv'
TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv'

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

train_images = {int(x.stem): x for x in Path(TRAIN_DIR).iterdir() if x.suffix == '.png'}
test_images = {int(x.stem): x for x in Path(TEST_DIR).iterdir() if x.suffix == '.png'}


size = 512  # 512

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
wrong_proteins_stats = ([0.0793, 0.0530, 0.0545], [0.1487, 0.1129, 0.1556])
proteins_stats = ([0.05438065, 0.05291743, 0.07920227], [0.39414383, 0.33547948, 0.38544176])

train_tfms = T.Compose([
    T.Resize(size),
    T.RandomCrop(size, padding=8, padding_mode='edge'),  # 512 if no size
    #     T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
    #     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.RandomHorizontalFlip(),
    T.RandomRotation(90),
    T.ToTensor(),
    T.Normalize(*proteins_stats, inplace=True),
    T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
    T.Resize(size),
    T.ToTensor(),
    T.Normalize(*proteins_stats)
])



NUM_WORKERS = 4

batch_size = 80
# resnet18: 80
# resnet34: 50
# resnet50: 20
# densenet121: 30

model_name = "resnet18"
NETWORK = getattr(models, model_name)(pretrained=True)

INTER_LAYER = 100
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


class HumanProteinDatasetOneHot(Dataset):
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


def create_split_df(data_df, nfolds=5, order=2):
    df = data_df.set_index("Image").sort_index()
    split_df = df.Label.str.split(" ").explode()
    split_df.value_counts()
    dummies_df = pd.get_dummies(split_df).groupby(split_df.index).sum()

    X, y = dummies_df.index.values, dummies_df.values
    k_fold = IterativeStratification(n_splits=nfolds, order=order)

    splits = list(k_fold.split(X, y))

    fold_splits = np.zeros(dummies_df.shape[0]).astype(np.int)

    for i in range(nfolds):
        fold_splits[splits[i][1]] = i

    dummies_df['Split'] = fold_splits

    df_folds = []

    for fold in range(nfolds):
        df_fold = dummies_df.copy()

        train_df = df_fold[df_fold.Split != fold].drop('Split', axis=1).reset_index()

        val_df = df_fold[df_fold.Split == fold].drop('Split', axis=1).reset_index()

        df_folds.append((train_df, val_df))

    return df_folds



def get_split_dataloaders(split):
    train_df, val_df = split

    train_ds = HumanProteinDatasetOneHot(train_df, transform=train_tfms)
    val_ds = HumanProteinDatasetOneHot(val_df, transform=valid_tfms)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size * 2, num_workers=NUM_WORKERS, pin_memory=True)

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    return train_dl, val_dl


def get_test_dl(submission):
    test_ds = HumanProteinDatasetOneHot(submission, transform=valid_tfms, is_test=True)
    test_dl = DataLoader(test_ds, batch_size*2, num_workers=NUM_WORKERS, pin_memory=True)
    return DeviceDataLoader(test_dl, device)
def focal_loss(out, targets, alpha=1, gamma=2, reduction='mean'):
    bce_loss = F.binary_cross_entropy(out, targets, reduction='none')

    # Prevents nans when probability 0
    pt = torch.exp(-bce_loss)

    F_loss = alpha * (1 - pt) ** gamma * bce_loss

    if reduction == "mean":
        return torch.mean(F_loss)
    elif reduction == "sum":
        return torch.sum(F_loss)
    else:
        return F_loss


def multi_label_loss(out, targets):
    return F.binary_cross_entropy(out, targets)
class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch, alpha=1, gamma=2):
        images, targets = batch
        out = self(images)
        loss = focal_loss(out, targets, alpha=alpha, gamma=gamma)
        return loss

    def validation_step(self, batch, threshold=0.5, alpha=1, gamma=2):
        images, targets = batch
        out = self(images)  # Generate predictions
        loss = focal_loss(out, targets, alpha=alpha, gamma=gamma)  # Calculate loss
        score = F_score(out, targets, threshold=threshold)
        return {'val_loss': loss.detach(), 'val_score': score.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}"
            .format(epoch, result['lrs'][-1], result['train_loss'],
                    result['val_loss'], result['val_score']))
        
        
class ProteinResnet(MultilabelImageClassificationBase):
    def __init__(self, resnet, inter_layer):
        super().__init__()
        # Use a pretrained model
        self.network = resnet
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        # self.network.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, inter_layer),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(inter_layer, 10),
        # )
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


class ProteinDensenet(MultilabelImageClassificationBase):
    def __init__(self, densenet):
        super().__init__()
        # Use a pretrained model
        self.network = densenet
        # Replace last layer
        num_ftrs = self.network.classifier.in_features
        self.network.classifier = nn.Linear(num_ftrs, 10)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.classifier.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True

@torch.no_grad()
def evaluate(model, val_loader, threshold=0.5, alpha=1, gamma=2):
    model.eval()
    outputs = [model.validation_step(batch, threshold=threshold, alpha=alpha, gamma=gamma) for batch in tqdm(val_loader)]
    return model.validation_epoch_end(outputs)


def fit_one_cycle(epochs,
                  max_lr,
                  model,
                  train_loader,
                  val_loader,
                  weight_decay=0,
                  grad_clip=None,
                  opt_func=torch.optim.Adam,
                  adam_betas=(0.9, 0.999),
                  adam_amsgrad=False,
                  scheduler=True,
                  threshold=0.5,
                  alpha=1,
                  gamma=2,
                  save_best="val_loss"):

    since = time.time()

    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay, betas=adam_betas, amsgrad=adam_amsgrad)

    if scheduler:
        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss, best_score = 1e4, 0.0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch, alpha=alpha, gamma=gamma)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))

            if scheduler:
                sched.step()

        # Validation phase
        result = evaluate(model, val_loader, threshold=threshold)
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

nfolds = 5

THRESHOLD = 0.3
ALPHA = 1
GAMMA = 2

freeze_unfreeze = True

epochs = 6
max_lr = 0.0005
sched_first_pass = True

second_epochs = 6
second_lr = 0.0005
sched_second_pass = True

grad_clip = 0.1
weight_decay = 1e-5
opt_func = torch.optim.AdamW
adam_betas = (0.9, 0.999)
adam_amsgrad = False


sub_fname = f'submission_{model_name}_{"sched" if sched_first_pass else ""}{epochs}_lr{max_lr}_' \
            f'{"sched" if sched_second_pass else ""}{second_epochs}_lr{second_lr}_crossv_{nfolds}.csv'
weights_fname = f'protein-{model_name}_{epochs}_lr{max_lr}_sched{second_epochs}_lr{second_lr}_crossv_{nfolds}.pth'
project_name = 'protein-advanced'

histories = []
predictions = []
since = time.time()

data_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
test_dl = get_test_dl(test_df)

splits = create_split_df(data_df, nfolds, order=2)

for i, split in enumerate(splits):
    history = []
    train_dl, val_dl = get_split_dataloaders(split)


    # If you want to try with densenets, you should replace ProteinResnet with ProteinDensenet
    model = to_device(ProteinResnet(NETWORK, inter_layer=INTER_LAYER), device)

    if freeze_unfreeze:
        model.freeze()

    model, hist = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func,
                             adam_betas=adam_betas,
                             adam_amsgrad=adam_amsgrad,
                             scheduler=sched_first_pass,
                             threshold=THRESHOLD,
                             alpha=ALPHA,
                             gamma=GAMMA)

    history += hist

    
    if freeze_unfreeze:
        model.unfreeze()

        
    if second_epochs > 0:
        model, hist = fit_one_cycle(second_epochs, second_lr, model, train_dl, val_dl,
                                    grad_clip=grad_clip,
                                    weight_decay=weight_decay,
                                    opt_func=opt_func,
                                    adam_betas=adam_betas,
                                    adam_amsgrad=adam_amsgrad,
                                    scheduler=sched_second_pass,
                                    threshold=THRESHOLD)

        history += hist

        
    test_preds = predict_dl_one_hot(test_dl, model, threshold=THRESHOLD)

    predictions.append(test_preds)

    del model
    gc.collect()

print(f'Total Training time: {(time.time() - since)/60:.2f} minutes')

prediction_cv = torch.stack(predictions).mean(axis=0)

submission_df = pd.read_csv(TEST_CSV)
submission_df.Label = [decode_target(t.tolist()) for t in prediction_cv] # test_preds


submission_df.to_csv(sub_fname, index=False)


# This is informative and will create the prediction CSVs for all the splits
for index, pred in enumerate(predictions):
    submission_df = pd.read_csv(TEST_CSV)
    submission_df.Label = [decode_target(t.tolist()) for t in pred]  # test_preds

    submission_df.to_csv(f"part{index}_" + sub_fname, index=False)
!pip install kaggle --upgrade
import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("KAGGLE_KEY")
secret_value_1 = user_secrets.get_secret("KAGGLE_USERNAME")

os.putenv("KAGGLE_KEY", user_secrets.get_secret("KAGGLE_KEY"))
os.putenv("KAGGLE_USERNAME", user_secrets.get_secret("KAGGLE_USERNAME"))
!kaggle competitions submit -f ./submission_resnet18_sched6_lr0.0005_sched6_lr0.0005_crossv_5.csv jovian-pytorch-z2g -m "public_notebook"