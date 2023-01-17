! pip install albumentations

! pip install pretrainedmodels
# libraries

import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import roc_auc_score

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler

import torch.backends.cudnn as cudnn

import time 

import tqdm

import random

from PIL import Image

train_on_gpu = True

from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR



import cv2



import albumentations

from albumentations import torch as AT

import pretrainedmodels



import scipy.special



cudnn.benchmark = True
SEED = 323

base_dir = '../input/'

def seed_everything(seed=SEED):

    random.seed(seed)

    os.environ['PYHTONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything(SEED)
labels = pd.read_csv(base_dir+'train_labels.csv')
print(len(os.listdir(base_dir+"train")))

print(len(os.listdir(base_dir+"test")))
fig = plt.figure(figsize=(25, 4))

# display 20 images

train_imgs = os.listdir(base_dir+"train")

for idx, img in enumerate(np.random.choice(train_imgs, 20)):

    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])

    im = Image.open(base_dir+"train/" + img)

    plt.imshow(im)

    lab = labels.loc[labels['id'] == img.split('.')[0], 'label'].values[0]

    ax.set_title('Label: %s'%lab)
labels.label.value_counts()
tr, val = train_test_split(labels.label, stratify=labels.label, test_size=0.15, random_state=SEED)
img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}
class CancerDataset(Dataset):

    def __init__(self, datafolder, datatype='train', transform = transforms.Compose([transforms.CenterCrop(64),transforms.ToTensor()]), labels_dict={}):

        self.datafolder = datafolder

        self.datatype = datatype

        self.image_files_list = [s for s in os.listdir(datafolder)]

        self.transform = transform

        self.labels_dict = labels_dict

        if self.datatype == 'train':

            self.labels = [labels_dict[i.split('.')[0]] for i in self.image_files_list]

        else:

            self.labels = [0 for _ in range(len(self.image_files_list))]



    def __len__(self):

        return len(self.image_files_list)



    def __getitem__(self, idx):

        img_name = os.path.join(self.datafolder, self.image_files_list[idx])

        img = cv2.imread(img_name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = self.transform(image=img)

        image = image['image']



        img_name_short = self.image_files_list[idx].split('.')[0]



        if self.datatype == 'train':

            label = self.labels_dict[img_name_short]

        else:

            label = 0

        return image, label
data_transforms = albumentations.Compose([

    #albumentations.CenterCrop(64, 64),

    albumentations.RandomRotate90(p=0.5),

    albumentations.Transpose(p=0.5),

    albumentations.Flip(p=0.5),

    albumentations.OneOf([

        albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(), 

        albumentations.RandomBrightness(), albumentations.RandomContrast(),

        albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 

    albumentations.HueSaturationValue(p=0.5), 

    albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),

    albumentations.Normalize(),

    AT.ToTensor()

    ])



data_transforms_test = albumentations.Compose([

    #albumentations.CenterCrop(64, 64),

    albumentations.Normalize(),

    AT.ToTensor()

    ])



data_transforms_tta0 = albumentations.Compose([

    #albumentations.CenterCrop(64, 64),

    albumentations.RandomRotate90(p=0.5),

    albumentations.Transpose(p=0.5),

    albumentations.Flip(p=0.5),

    albumentations.Normalize(),

    AT.ToTensor()

    ])



data_transforms_tta1 = albumentations.Compose([

    #albumentations.CenterCrop(64, 64),

    albumentations.RandomRotate90(p=1),

    albumentations.Normalize(),

    AT.ToTensor()

    ])



data_transforms_tta2 = albumentations.Compose([

    #albumentations.CenterCrop(64, 64),

    albumentations.Transpose(p=1),

    albumentations.Normalize(),

    AT.ToTensor()

    ])



data_transforms_tta3 = albumentations.Compose([

    #albumentations.CenterCrop(64, 64),

    albumentations.Flip(p=1),

    albumentations.Normalize(),

    AT.ToTensor()

    ])



dataset = CancerDataset(datafolder=base_dir+'train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)

val_set = CancerDataset(datafolder=base_dir+'train/', datatype='train', transform=data_transforms_test, labels_dict=img_class_dict)

test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_test)

train_sampler = SubsetRandomSampler(list(tr.index)) 

valid_sampler = SubsetRandomSampler(list(val.index))

batch_size = 256

num_workers = 0

# prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
model_conv = pretrainedmodels.se_resnext50_32x4d()

model_conv.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

model_conv.last_linear = nn.Linear(in_features=2048, out_features=1, bias=True)
# class FocalLoss(nn.Module):

#     def __init__(self, alpha=0.42, gamma=2, logits=False, reduce=True):

#         super(FocalLoss, self).__init__()

#         self.alpha = alpha

#         self.gamma = gamma

#         self.logits = logits

#         self.reduce = reduce



#     def forward(self, inputs, targets):

#         if self.logits:

#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)

#         else:

#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

#         pt = torch.exp(-BCE_loss)

#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss



#         if self.reduce:

#             return torch.mean(F_loss)

#         else:

#             return F_loss
model_conv.cuda()

criterion = nn.BCEWithLogitsLoss()

#criterion = FocalLoss()



optimizer = optim.Adam(model_conv.parameters(), lr=0.00005)

scheduler = StepLR(optimizer, 1, gamma=0.5)
valid_loss_min = np.Inf

patience = 5

# current number of tests, where validation loss didn't increase

p = 0

# whether training should be stopped

stop = False



# number of epochs to train the model

n_epochs = 6

for epoch in range(1, n_epochs+1):

    print(time.ctime(), 'Epoch:', epoch)

    

    if stop:

        print("Training stop.")

        break



    train_loss = []

    scheduler.step(epoch)

    train_auc = []



    for tr_batch_i, (data, target) in enumerate(train_loader):

        model_conv.train()

        #time_s = time.time()



        data, target = data.cuda(), target.cuda()



        optimizer.zero_grad()

        output = model_conv(data)

        loss = criterion(output[:,0], target.float())

        train_loss.append(loss.item())

        

        a = target.data.cpu().numpy()

        b = output[:,0].detach().cpu().numpy()

        train_auc.append(roc_auc_score(a, b))



        loss.backward()

        optimizer.step()

        

        #time_e = time.time()

        #delta_t = time_e - time_s

        #print("training.... (time cost:%.3f s)"% delta_t)

        

        if (tr_batch_i+1)%100 == 0:    

            model_conv.eval()

            val_loss = []

            val_auc = []

            for val_batch_i, (data, target) in enumerate(valid_loader):

                data, target = data.cuda(), target.cuda()

                output = model_conv(data)



                loss = criterion(output[:,0], target.float())



                val_loss.append(loss.item()) 

                a = target.data.cpu().numpy()

                b = output[:,0].detach().cpu().numpy()

                val_auc.append(roc_auc_score(a, b))



            # print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}, train auc: {np.mean(train_auc):.4f}, valid auc: {np.mean(val_auc):.4f}')

            print('Epoch %d, batches:%d, train loss: %.4f, valid loss: %.4f.'%(epoch, tr_batch_i, np.mean(train_loss), np.mean(val_loss)) +

                  '  train auc: %.4f, valid auc: %.4f'%(np.mean(train_auc),np.mean(val_auc)))

            train_loss = []

            train_auc = []

            valid_loss = np.mean(val_loss)

            if valid_loss <= valid_loss_min:

                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

                valid_loss_min,

                valid_loss))

                #torch.save(model_conv.state_dict(), 'save/model_valloss%d.pt'%(valid_loss*10000))

                torch.save(model_conv.state_dict(), 'model.pt')

                valid_loss_min = valid_loss

                p = 0

            else:

                p += 1

            if p > patience:

                print('Early stop training')

                stop = True

                break          
torch.cuda.empty_cache()
model_conv.eval()
saved_dict = torch.load('model.pt')

model_conv.load_state_dict(saved_dict)

preds = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.cuda(), target.cuda()

    output = model_conv(data).detach()



    pr = output[:,0].cpu().numpy()

    for i in pr:

        preds.append(i)
test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})

test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])

sub = pd.read_csv('../input/sample_submission.csv')

sub = pd.merge(sub, test_preds, left_on='id', right_on='imgs')

sub = sub[['id', 'preds']]

sub.columns = ['id', 'label']

sub.head()
sub.to_csv('sub.csv', index=False)
NUM_TTA = 8
sigmoid = lambda x: scipy.special.expit(x)
for num_tta in range(NUM_TTA):

    if num_tta==0:

        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_test)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    elif num_tta==1:

        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_tta1)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    elif num_tta==2:

        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_tta2)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    elif num_tta==3:

        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_tta3)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    else:

        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_tta0)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    

    preds = []

    for batch_i, (data, target) in enumerate(test_loader):

        data, target = data.cuda(), target.cuda()

        output = model_conv(data).detach()

        pr = output[:,0].cpu().numpy()

        for i in pr:

            preds.append(sigmoid(i)/NUM_TTA)

    if num_tta==0:

        test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})

        test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])

    else:

        test_preds['preds']+=np.array(preds)

    

sub = pd.read_csv('../input/sample_submission.csv')

sub = pd.merge(sub, test_preds, left_on='id', right_on='imgs')

sub = sub[['id', 'preds']]

sub.columns = ['id', 'label']

sub.to_csv('sub_tta.csv', index=False)
print("Finished")