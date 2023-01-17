import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from keras.utils import to_categorical

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as T
from torchvision import models
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
lesion_categorical = {k:i for i, k in enumerate(lesion_type_dict)}
DATA_DIR = '../input/skin-cancer-mnist-ham10000'
original_df = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))
original_df.head()
original_df['localization'].unique()
localization_to_index = {name:i for i, name in enumerate(original_df['localization'].unique())}
original_df['lesion_id'].nunique()
# Drop dx_type since we don't care about it when predicting dx
temp_df = original_df.drop('dx_type', axis = 1)
# Convert dx to categorical
temp_df['dx'] = temp_df['dx'].map(lesion_categorical)
# Drop null values
temp_df = temp_df.dropna()
temp_df.head()
freqs = temp_df['dx'].value_counts()
freqs
upsample_rate = [6705 // freq for freq in freqs]

for i in range(1, 7):
    temp_df = temp_df.append([temp_df.loc[temp_df['dx'] == i,:]]*(upsample_rate[i]-1), ignore_index=True)
    
temp_df['dx'].value_counts(normalize=True)
#duplicates = temp_df.groupby('lesion_id')['image_id'].count()
#duplicates.head()
#temp_df['has_duplicates'] = temp_df['lesion_id'].apply(
#    lambda lesion_id: duplicates[lesion_id] > 1
#)
#temp_df['has_duplicates'].value_counts()
#train_df = temp_df[temp_df['has_duplicates']]
#mixed_df = temp_df[~temp_df['has_duplicates']]
#train_df['dx'].value_counts(normalize=True)
#val_df['dx'].value_counts(normalize=True)
#mixed_df.shape[0], train_df.shape[0]
temp_df.shape[0]
train_df, val_df = train_test_split(temp_df, test_size=0.2)
train_df.shape[0], val_df.shape[0]
#train_df = pd.concat([train_df, train_temp], ignore_index=True)
#train_df.shape[0], val_df.shape[0]
# Drop the has_duplicates column since we don't need it anymore
#train_df = train_df.drop('has_duplicates', axis = 1)
#val_df = val_df.drop('has_duplicates', axis = 1).reset_index()
train_df.reset_index(inplace=True, drop=True)
val_df.reset_index(inplace=True, drop=True)
train_df.head()
val_df.head()
# https://www.kaggle.com/xinruizhuang/skin-lesion-classification-acc-90-pytorch
mean = [0.763038, 0.54564667, 0.57004464]
std = [0.14092727, 0.15261286, 0.1699712]
class SkinCancerDataset(Dataset):
    def __init__(self, df, transforms = None):
        self.df = df
        self.transforms = transforms
        self.part1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
        self.part2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image_id = row['image_id']
        dx = row['dx']
        features = list(row.loc['age':'localization'])
        age, sex, localization = features[0], features[1], features[2]
        sex = 0 if sex == 'male' else 1
        localization = localization_to_index[localization]
        sex = to_categorical(sex, num_classes = 2)
        localization = to_categorical(localization, num_classes = len(localization_to_index))
        
        try:
            img = Image.open(os.path.join(self.part1, image_id + '.jpg'))
        except FileNotFoundError:
            img = Image.open(os.path.join(self.part2, image_id + '.jpg'))
            
        if self.transforms != None:
            img = self.transforms(img)

        return  torch.tensor([age], dtype=torch.float32), torch.tensor(sex, dtype=torch.float32),\
                torch.tensor(localization, dtype=torch.float32), img, dx
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
device = get_default_device()
device
# feature_extract is a boolean that defines if we are finetuning or feature extracting. 
# If feature_extract = False, the model is finetuned and all model parameters are updated. 
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    
    assert model_name in ['resnet', 'vgg', 'densenet', 'inception'], 'Invalid model name'

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101 """
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "densenet":
        """ Densenet121 """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    return model_ft, input_size
model_name = 'densenet'
model_ft, image_size = initialize_model(model_name, 20, use_pretrained=True)
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        age, sex, localization, image, label = batch
        out = self(image, age, sex, localization) 
        loss = F.cross_entropy(out, label)      
        return loss
    
    def validation_step(self, batch):
        age, sex, localization, image, label = batch
        out = self(image, age, sex, localization)
        loss = F.cross_entropy(out, label)
        score = F_score(out, label)
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

class MyModel(ImageClassificationBase):
    def __init__(self, model_name):
        super().__init__()
        self.cnn = model_ft
        self.fc1 = nn.Linear(20 + 1 + 2 + 15, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 7)
        
    def forward(self, image, age, sex, localization):
        x1 = self.cnn(image)
        x = torch.cat([x1, age, sex, localization], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
train_ds = SkinCancerDataset(train_df, transforms=T.Compose([
                                                                T.RandomCrop(image_size, padding=8, padding_mode='reflect'),
                                                                T.ToTensor(),
                                                                T.Normalize(mean, std, inplace=True)
                                                            ])
                            )
val_ds = SkinCancerDataset(val_df, transforms=T.Compose([
                                                                T.RandomCrop(image_size, padding=8, padding_mode='reflect'),
                                                                T.ToTensor(),
                                                                T.Normalize(mean, std, inplace=True)
                                                            ])
                            )
batch_size = 32

train_dl = DataLoader(train_ds, batch_size, shuffle=True, 
                      num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, 
                    num_workers=2, pin_memory=True)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
model = MyModel(model_name)
model = model.to(device)
batch_size = 1
image = torch.randn(batch_size, 3, 224, 224).to(device)
age = torch.randn(batch_size, 1).to(device)
sex = torch.randn(batch_size, 2).to(device)
localization = torch.randn(batch_size, 15).to(device)
model(image, age, sex, localization)
def F_score(output, label, beta=1):
    _, prob = output.max(dim=1)
    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, freeze, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=optim.Adam):
    model.train()
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    # Freeze or Unfreeze
    set_parameter_requires_grad(model, freeze)
    for epoch in range(epochs):
        # Training Phase 
        
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
# Train all layers
model = MyModel(model_name)
model = model.to(device)
epochs = 10
max_lr = 3e-3
freeze = False
history = fit_one_cycle(epochs, 
                        max_lr,
                        model,
                        freeze, 
                        train_dl, 
                        val_dl)
# Train all layers
#history += fit_one_cycle(epochs,
#                         max_lr/10,
#                         model,
#                         freeze,
#                         train_dl,
#                         val_dl)