import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from PIL import Image

from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
# get access to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# use cpu for preprocessing and setting up the initial model to test
device
SEED=42
# Load Data from csv and images
class Data():
    def __init__(self, base, resized):
        self.base = '/kaggle/input/' + base
        self.resized = '/kaggle/input/' + resized
        
    def b(self, path):
        return self.base + path
    
    def r(self, path):
        return self.resized + path
    
    def read(self, file):
        return pd.read_csv(self.b(file))
    
data = Data(base='siim-isic-melanoma-classification/', resized='jpeg-melanoma-256x256/')

df_train = data.read('train.csv')
df_test = data.read('test.csv')
df_submit = data.read('sample_submission.csv')
# patients who were diagnosed
df_train[df_train['target']==1]
df_train['diagnosis'].value_counts()
def get_new_size(width, height, minimum, width_first=True):
    h, w = 0, 0
    if width >= height:
        w = round(minimum * (width / height))
        h = minimum
    else:
        w = minimum
        h = round(minimum * (height / width))
    if width_first:
        return w, h
    else:
        return h, w

print(get_new_size(6000, 40000, 256))
print(get_new_size(6000, 40000, 256, width_first=False))
# Setup a grid to show random images from subset
def show_grid(df, cols=9, rows=4):
    if df.shape[0] == 0:
        return
    plt.figure(figsize=(18,9))
    for i in range(min(df.shape[0], cols * rows)):
        plt.subplot(rows, cols, i+1, xticks=[], yticks=[])
        idx = np.random.randint(0, df.shape[0], 1)[0]
        im = Image.open(data.r('train/' + df.iloc[idx]['image_name'] + '.jpg'))
        plt.imshow(im)
        plt.xlabel(df.iloc[idx]['benign_malignant'])
        plt.ylabel(df.iloc[idx]['anatom_site_general_challenge'])
    plt.show()

# Check young people
# show_grid(df_train[(df_train['age_approx'] < 40.0) & (df_train['target'] == 1)])
# Check diagnosis
# show_grid(df_train[(df_train['diagnosis'] == 'melanoma') & (df_train['target'] == 1)])
# Check a single patient
pat = 'IP_0962375'
show_grid(df_train[(df_train['patient_id'] == pat) & (df_train['target'] == 1)])
show_grid(df_train[(df_train['patient_id'] == pat) & (df_train['target'] == 0)])

# Mark male female as 1/0
# There are only two values, there are some missing values, which should be filled with mode
df_train['sex'] = df_train['sex'].replace({ 'female': 0, 'male': 1 })
df_test['sex'] = df_test['sex'].replace({ 'female': 0, 'male': 1 })
df_train['sex'].fillna(df_train['sex'].mode()[0], inplace=True)

# Remove benign malignant, it's the same as target
df_train.drop(['benign_malignant'], inplace=True, axis=1)

# Add dummies for anatom_site_general_challenge
# Fill the nan's with a new dummy
def add_dummies(dataset, column, short_name):
    dummy = pd.get_dummies(
        dataset[column], 
        drop_first=True, 
        prefix=short_name, 
        prefix_sep='_',
        dummy_na=True
    )
    merged = pd.concat([dataset, dummy], axis=1)
    return merged.drop([column], axis=1)

df_train = add_dummies(df_train, 'anatom_site_general_challenge', 'anatom')
df_test = add_dummies(df_test, 'anatom_site_general_challenge', 'anatom')

# Diagnosis is only in train, removing it
df_train.drop(['diagnosis'], inplace=True, axis=1)

# Age has some missing values, fill with median
df_train['age_approx'].fillna(df_train['age_approx'].median(), inplace=True)

# %% [code]
# Check how many times are their images taken
df_train['image_count'] = df_train['patient_id'].map(df_train.groupby(['patient_id'])['image_name'].count())
df_test['image_count'] = df_test['patient_id'].map(df_test.groupby(['patient_id'])['image_name'].count())
df_train
df_test
# How does one scale in pytorch :(

sc = StandardScaler()

def scale(df, cols_to_remove, fit=False):
    removed = df[cols_to_remove]
    df = df.drop(cols_to_remove, axis=1)
    cols = df.columns
    if fit:
        df = sc.fit_transform(df)
    else:
        df = sc.transform(df)
    df = pd.DataFrame(df, columns=cols)
    df[cols_to_remove] = removed
    return df

df_train = scale(df_train, fit=True, cols_to_remove=['image_name', 'patient_id', 'target'])
df_test = scale(df_test, cols_to_remove=['image_name', 'patient_id'])
# Experiment with pytorch transforms

# ## read image
# size = (256, 256)
# im_path = get_path('jpeg/train/ISIC_0015719.jpg')
# start = time.time()
# im = Image.open(im_path)
# print('Image.open: ', time.time() - start)
# # print(im.format, im.size, im.mode, type(im))
# # print(im.size[0], im.size[1])
# # print(im.format, im.size, im.mode)

# # im = im.resize(get_new_size(im.size[0], im.size[1], 256))
# start = time.time()
# im = transforms.Resize(get_new_size(im.size[0], im.size[1], 256, width_first=False))(im)
# print('Resize: ', time.time() - start)
# start = time.time()
# im = transforms.CenterCrop(256)(im)
# print('CenterCrop: ', time.time() - start)
# plt.imshow(im)
# plt.show()

# # im = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)(im)
# # im = transforms.Grayscale(1)(im)
# # im = transforms.Pad(64, fill=(255, 255, 255), padding_mode='symmetric')(im)
# # im2 = transforms.RandomAffine(degrees=0, translate=(0, 0), scale=None, shear=None)(im) # , scale=(1, 0.1)
# # im2 = transforms.RandomApply([transforms.Grayscale(1)], 0.5)(im)
# # im2 = transforms.RandomChoice([
# #     transforms.Grayscale(1),
# #     transforms.Pad(64, fill=(255, 255, 255), padding_mode='symmetric'),
# # ])(im)
# # im2 = transforms.RandomHorizontalFlip(0.99)(im)
# # im2 = transforms.RandomOrder([
# #     transforms.Grayscale(1),
# #     transforms.Pad(64, fill=(255, 255, 255), padding_mode='symmetric'),
# # ])(im)
# # im2 = transforms.RandomPerspective(distortion_scale=0.5)(im)
# # im2 = transforms.RandomVerticalFlip(0.99)(im)
# # im2 = transforms.LinearTransformation(transformation_matrix=[0,0,0], mean_vector=[0,0,0])(im) # not sure
# # im2 = transforms.ToPILImage()(
# #     transforms.RandomErasing(p=0.99)(
# #         transforms.ToTensor()(im)
# #     )
# # )
# # im2 = transforms.functional.adjust_brightness(im, brightness_factor=2)
# # im2 = transforms.functional.adjust_contrast(im, contrast_factor=3)
# start = time.time()
# im2 = transforms.functional.adjust_gamma(im, gamma=0.5)
# print('adjust_gamma: ', time.time() - start)
# plt.imshow(im2)
# plt.show()
# Visualise data by segmenting each parameter
# Find and note down outliers
SIZE=128

class PatientImages(Dataset):
    
    def __init__(self, df, is_training=True, augment=False, debug=False):
        super(PatientImages, self).__init__()
        self.is_training = is_training
        self.augment = augment
        self.debug = debug

        if self.is_training:
            self.x = df.drop(['target'], axis=1).values
            self.y = df['target'].values
        else:
            self.x = df.values
        self.columns = df.columns
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = None
        if self.is_training:
            y = self.y[idx]
        
        img = x[9]
        train_or_test = 'train' if self.is_training == True else 'test'
        img_path = data.r(train_or_test + '/' + img + '.jpg')
        img = Image.open(img_path)

        if self.debug:
            plt.imshow(img)
            plt.show()

        adjusted_size = self.get_adjusted_size(img.size[0], img.size[1], SIZE)

        transform = transforms.Compose([
            # transforms.Resize(adjusted_size), # Using https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256
            # transforms.CenterCrop(SIZE), # Using https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256
            # transforms.ToTensor(), # Applied manually
        ])

        if self.augment:
            transform = transforms.Compose([
                # transforms.Resize(adjusted_size), # Using https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256
                # transforms.CenterCrop(SIZE), # Using https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.25),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomAffine(degrees=10, shear=2),
                transforms.RandomPerspective(p=0.05, distortion_scale=0.1),
                transforms.ToTensor(),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random'),
                transforms.ToPILImage(),
                # transforms.ToTensor(), # Applied manually
            ])
            
        img = transform(img)
        if self.debug:
            plt.imshow(img)
            plt.show()
        x_img = transforms.ToTensor()(img)

        x = x[:-2]
                    
        if type(x).__module__ == 'numpy':
            x = x.astype(np.float32)
            
        if self.is_training:
            return (x, x_img), y
        else:
            return (x, x_img)
    
    def __len__(self):
        return self.x.shape[0]
    
    def columns(self):
        return self.columns
    
    def get_adjusted_size(self, width, height, minimum):
        h, w = 0, 0
        if width >= height:
            w = round(minimum * (width / height))
            h = minimum
        else:
            w = minimum
            h = round(minimum * (height / width))
            
        # if self.debug:
            # print(f'Original: h{height} w{width} m{minimum} - Calculated: h{h} w{w}')
        return h, w
# Test Dataset
start = time.time()
train_test = PatientImages(df_train, augment=True, debug=True)
print(train_test[50])
print(train_test.__len__())
end = time.time()
print('Timed', end - start)

start = time.time()
test_test = PatientImages(df_test, is_training=False, debug=True)
print(test_test[50])
print(test_test.__len__())
end = time.time()
print('Timed', end - start)
BATCH_SIZE=64

start = time.time()
train_ds = PatientImages(df_train)
test_ds = PatientImages(df_test, is_training=False)
end = time.time()
print('Timed', end-start)

start = time.time()
train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True)
end = time.time()
print('Timed', end-start)
# Test Dataloaders
start = time.time()
print(train_ds)
print(test_ds)
end = time.time()
print('Timed _ds', end-start)

start = time.time()
print(train_ds[0])
print(test_ds[0])
end = time.time()
print('Timed _ds[0]', end-start)

start = time.time()
print(len(train_dl))
print(len(test_dl))
end = time.time()
print('Timed _dl', end-start)

start = time.time()
features, label = iter(train_dl).next()
print(features[0].shape, features[1].shape, label.shape)
end = time.time()
print('Timed train_images_dl[0]', end-start)

start = time.time()
features = iter(test_dl).next()
print(features[0].shape, features[1].shape)
end = time.time()
print('Timed test_images_dl[0]', end-start)
# Start with dumb pipeline
## Start with a dumb linear regression model
## Define how to merge data from csv files
## Setup initial bias properly for layers
## Verify loss at start, should be equal to -log(1/n_classes) on softmax
## Setup an input with all 0s to test model output
## overfit one batch with 2 samples, check if loss is 0
## As you add to the model, loss should go down
## Visualise data right before it goes into the model before model(x)
## Visualise prediction dynamics
# Overfit
## Start with standard models
## Use adam, starting with lr 3e-4
## add complexity one at a time, add signals later
## change the learning rate decay, or keep it at 0, let me the model converge
# Regularise
## visualise the first layer weights
## augment data
## add ensembles
## use pre-trained models
## keep the batch size low
## add dropout or dropout2d
## increase weight decay penalty
## early stopping
## last try a larger model
# Tune
## random over grid search
## hyper parameter optimisation
# Squeeze
## ensembles
## leave it training
!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

# Setup the model
class NN(nn.Module):
    
    def __init__(self):
        super(NN, self).__init__()

        self.efn = EfficientNet.from_pretrained('efficientnet-b1')
        self.efn._fc = nn.Linear(1280, 500, bias=True)
        
        self.meta = nn.Sequential(
            nn.Linear(9, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
        self.output = nn.Linear(500 + 250, 1)

        
    def forward(self, x):
        meta, images = x
        cnn = self.efn(images)
        others = self.meta(meta)
        features = torch.cat((cnn, others), dim=1)
        output = self.output(features)
        return output

learning_rate = 1e-2
model = NN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
# torch.cuda.empty_cache()

# Setup the training loop
n_epochs = 3
total_steps = len(train_dl)

pbar = tqdm(total=len(train_dl), desc='Epoch: 0, Loss: 0.000000')
for epoch in range(n_epochs):
    pbar.reset()
    for i, (dl_data, labels) in enumerate(train_dl):
        start = time.time()
        meta, images = dl_data
        
        meta = meta.to(device)
        images = images.to(device)
        labels = labels.to(device)
            
        labels = labels.view(labels.shape[0], -1).float()
        
        model.train()
        outputs = model((meta, images))
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        
        pbar.update(1)
        pbar.set_description(f'Epoch: {epoch+1}, Loss: {loss.item():.6f}')
        
pbar.close()
# [ (name, param.data) for name, param in model.named_parameters()]
torch.save(model, 'model.pth')
# Setup the evaluation functions

with torch.no_grad():
    count = 0
    op_ = None
    lb_ = None
    
    pbar = tqdm(total=len(train_dl), desc='Accuracy: 00.0000%')

    start = time.time()
    for dl_data, labels in train_dl:
        meta, images = dl_data
        meta = meta.to(device)
        images = images.to(device)
        labels = labels.to(device)
        
        model.eval()
        output = model((meta, images))
        output = torch.sigmoid(output)
        output = output.squeeze()
        
        op_ = output if op_ == None else torch.cat((op_, output), 0)        
        lb_ = labels if lb_ == None else torch.cat((lb_, labels), 0)

        count += 1
        pbar.update(1)
    
        op = op_.cpu()
        op = op.round().long()
        lb = lb_.cpu()
        lb = lb
        
        pbar.set_postfix(
            { 
                'f1': 100*f1_score(lb, op, average='weighted'), 
                'logloss': log_loss(lb, op, labels=[0, 1])
            }
        )
        pbar.set_description(f'Accuracy: {accuracy_score(lb, op)*100:.4f}%')
            
    pbar.close()
    print('Timed:', time.time() - start)
            
    print('CM', confusion_matrix(lb, op))
    print('CR', classification_report(lb, op))
    print(precision_recall_fscore_support(lb, op))
    print('PRCurve', precision_recall_curve(lb, op))
    print('Pri', precision_score(lb, op))
    print('Rec', recall_score(lb, op))
    print('ROC', roc_auc_score(lb, op))
