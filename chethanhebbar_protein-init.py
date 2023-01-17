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
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
%matplotlib inline
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'

TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images
TEST_DIR = DATA_DIR + '/test'                             # Contains test images

TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images
TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv'   # Contains dummy labels for test image
# creating a labels dict to help us out in the future
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
# lets create a pandas dataframe to simplify our work on train_csv
data_df = pd.read_csv(TRAIN_CSV)
data_df.info()
data_df.head()
# we create to functions to convert our string label to a tensor label for the model to work with

def encode_label(label):
    target = torch.zeros(10)
    for l in str(label).split(' '):
        target[int(l)] = 1.
    return target
# we create a function to convert this tensor back into a string label

def decode_target(target, text_labels=False, threshold=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)
# lets try out these functions 
# indexing from 0
encode_label("5 8 1")
decode_target(torch.tensor([0,1,0,1,0,1,0,1,0,1]))
decode_target(torch.tensor([0,1,0,1,0,1,0,1,0,1]), text_labels = True)
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
        img_fname = self.root_dir + "/" + str(img_id) + ".png"
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)
        return img, encode_label(img_label)
stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = T.Compose([T.RandomCrop(512, padding = 4, padding_mode='reflect'),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(*stats,inplace = True),T.RandomErasing(inplace = True)])

val_transform = T.Compose([T.ToTensor(),T.Normalize(*stats)])
np.random.seed(42)
msk = np.random.rand(len(data_df)) < 0.8

train_df = data_df[msk].reset_index()
val_df = data_df[~msk].reset_index()
train_ds = HumanProteinDataset(train_df, TRAIN_DIR, transform = train_transform)
val_ds = HumanProteinDataset(val_df, TRAIN_DIR, transform = val_transform)

len(train_ds), len(val_ds)
# now lets look at one of the images from the dataset
def show_sample(img, target, invert = True):
    
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
        
    print('Labels:', decode_target(target, text_labels=True))
    
# we change the indexing from 0,1,2 to 1,2,0 to plot using matplotlib
show_sample(*train_ds[0], invert = False)
show_sample(*train_ds[0])
#torch.manual_seed(10)
# using a validation percentage of 20%
#val = 0.2
#val_size = int(val * len(dataset))
#train_size = len(dataset) - val_size
#train_ds, val_ds = random_split(dataset, [train_size, val_size])#
#len(train_ds), len(val_ds)
# declaring batch size
batch_size = 32
# forming the train and validation dataloaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers = 2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 2, pin_memory=True)
def show_batch(dl, invert=True):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        break
show_batch(train_dl, invert = False)
# defining the metrics 

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
# creating the base model

class MultiImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        
        images, targets = batch 
        out = self(images)
        
        # using binary cross entropy loss function...
        loss = F.binary_cross_entropy(out, targets)      
        return loss
    
    def validation_step(self, batch):
        
        images, targets = batch 
        out = self(images)       # Generate predictions
        
        # using binary cross-entropy loss function
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
class ProteinCNN(MultiImageClassificationBase):
    
    # constructor function
    def __init__(self):
        
        super().__init__()
        
        # creating the network
        self.network = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(15, 60, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(60, 120, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(120, 90, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(90, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(30, 15, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(), 
            nn.Linear(240, 80),
            nn.ReLU(),
            
            nn.Linear(80, 10),
            nn.Softmax()
        )
        
    def forward(self, xb):
        
        return self.network(xb)
class ProteinCNN2(MultiImageClassificationBase):
    
    def __init__(self):
        
        super().__init__()
        
        # Use a pretrained model
        self.network = models.resnet34(pretrained = True)
        
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
model = ProteinCNN2()
model
def get_default_device():
    #Pick GPU if available, else CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    
    #Move tensor(s) to chosen device
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    
    #Wrap a dataloader to move data to a device
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        #Yield a batch of data after moving it to device
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        #Number of batches
        return len(self.dl)
# getting the gpu 
device = get_default_device()
device
# transferring the dataloaders to the GPU
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)
# making random predictions
def try_batch(dl):
    for images, labels in dl:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[0])
        break

try_batch(train_dl)
# loads the epochs...
from tqdm.notebook import tqdm
# creating the evaluate and fit functions...

def evaluate(model, val_loader):
    
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, max_lr, model, train_loader, val_loader, decay, opt_func, grad_clip = None):
    
    torch.cuda.empty_cache()
    history = []
    
    #setting up the optimizer
    optimizer = opt_func(model.parameters(), max_lr, weight_decay = decay)
    
    # setting up the learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs = epochs, steps_per_epoch = len(train_loader))
    
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
        history.append(result)
        
    return history
# replacing the model into the gpu
model = to_device(ProteinCNN2(), device)
torch.cuda.empty_cache()
# evaluating the model which is randomized
evaluate(model, val_dl)
# freezing the model and training some epochs
model.freeze()
# setting up the hyperparams
num_epochs = 1
opt_func = torch.optim.Adamax
grad_clip = 1e-1
max_lr = 1e-2
decay = 1e-4
# training the model
torch.cuda.empty_cache()
history += fit(num_epochs, max_lr, model, train_dl, val_dl, decay, opt_func, grad_clip)
# training the model
torch.cuda.empty_cache()
history += fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
# function to predict a single image

def predict_single(image):
    
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    prediction = preds[0]
    print("Prediction: ", prediction)
    show_sample(image, prediction)
# loading the test dataset
test_dataset = HumanProteinDataset(TEST_CSV, TEST_DIR, transform=transform)
# looking at one of the images in the test dataset
img, target = test_dataset[0]
img.shape
# prediction on discrete images
predict_single(test_dataset[100][0])
predict_single(test_dataset[74][0])
# getting the dataloader from the dataset
test_dl = DeviceDataLoader(DataLoader(test_dataset, batch_size, num_workers=2, pin_memory=True), device)
# getting the predictions for the entire test data loader
def predict_dl(dl, model):
    
    torch.cuda.empty_cache()
    batch_probs = []
    
    for xb, _ in tqdm(dl):
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
        
    batch_probs = torch.cat(batch_probs)
    return [decode_target(x) for x in batch_probs]
test_preds = predict_dl(test_dl, model)
# updating submission.csv
submission_df = pd.read_csv(TEST_CSV)
submission_df.Label = test_preds
submission_df.head()
sub_fname = 'resnet34_submission_opt.csv'
# storing our submission into a .csv file
submission_df.to_csv(sub_fname, index=False)
!pip install jovian --upgrade

import jovian

jovian.commit(project='zerogans-protein-competition')