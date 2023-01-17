# Install pytorchcv

!pip install ../input/pytorchcv/pytorchcv-0.0.55-py2.py3-none-any.whl --quiet
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm,trange

from sklearn.model_selection import train_test_split

import sklearn.metrics



import torch

import torch.nn as nn

import torch.nn.functional as F



import warnings

warnings.filterwarnings("ignore")



from sklearn.metrics import cohen_kappa_score



def quadratic_weighted_kappa(y_hat, y):

    return cohen_kappa_score(y_hat, y, weights='quadratic')



class config:

    IMAGE_WIDTH = 512

    IMAGE_HEIGHT = 515

    epochs = 10

    batch_size = 16

    num_classes = 6

    IMAGE_PATH = '../input/panda-resized-train-data-512x512/train_images/train_images/'

    lr = 1e-4
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

mskf = StratifiedKFold(n_splits=5, random_state=12)



train_df2 = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

train_df2 = train_df2.drop(['gleason_score'], axis=1)

X, y = train_df2.values[:,0:2], train_df2[['isup_grade']].values[:,0]



train_df2['fold'] = -1

for fld, (_, test_idx) in enumerate(mskf.split(X, y)):

    train_df2.iloc[test_idx, -1] = fld
from torch.utils.data import Dataset, DataLoader

from PIL import Image



class ImageDataset(Dataset):

    def __init__(self, dataframe, root_dir, folds, transform=None):

        self.df = dataframe[dataframe.fold.isin(folds).reset_index(drop=True)]

        self.root_dir = root_dir

        self.transform = transform

        self.folds = folds



        self.paths = self.df.image_id.values

        self.labels = self.df.values[:,2]



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        img_name = self.paths[idx]

        img_path = f'{self.root_dir}{img_name}.png'

        

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

        img = (255 - img).astype(np.float32) / 255.



        if self.transform is not None:

          img = self.transform(image=img)['image']

        

        img = np.rollaxis(img, -1, 0)

        

        labels = np.array(self.labels[idx]).astype(np.long)

        return [img, labels]
from pytorchcv.model_provider import get_model



class Head(torch.nn.Module):

  def __init__(self, in_f, out_f, dropout):

    super(Head, self).__init__()

    

    self.f = nn.Flatten()

    self.d = nn.Dropout(0.25)

    self.dropout = dropout

    self.o = nn.Linear(in_f, out_f)



  def forward(self, x):

    x = self.f(x)

    if self.dropout:

      x = self.d(x)



    out = self.o(x)

    return out



class FCN(torch.nn.Module):

  def __init__(self, base, in_f, num_classes, dropout=True):

    super(FCN, self).__init__()

    self.base = base

    self.h1 = Head(in_f, num_classes, dropout)

  

  def forward(self, x):

    x = self.base(x)

    return self.h1(x)



def create_model():

    model = get_model("seresnext50_32x4d", pretrained=False)

    model.load_state_dict(torch.load('../input/seresnext50-32x4d-pretrained/seresnext50_32x4d-0521-b0ce2520.pth'))

    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer

    model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))

    model = FCN(model, 2048, config.num_classes, dropout=True)

    return model
# Ref: https://www.kaggle.com/c/bengaliai-cv19/discussion/128637



def mixup(data, targets1, alpha):

    indices = torch.randperm(data.size(0))

    shuffled_data = data[indices]

    shuffled_targets1 = targets1[indices]



    lam = np.random.beta(alpha, alpha)

    data = data * lam + shuffled_data * (1 - lam)

    targets = [targets1, shuffled_targets1, lam]



    return data, targets



def mixup_criterion(preds1, targets):

    targets1, targets2, lam = targets[0], targets[1], targets[2]

    criterion = nn.CrossEntropyLoss(reduction='mean')

    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)
def criterion1(pred1, targets):

  l1 = F.cross_entropy(pred1, targets)

  return l1



def train_model(epoch, optimizer, scheduler=None, history=None):

    model.train()

    total_loss = 0

    

    t = tqdm(train_loader)

    for batch_idx, (img_batch, y_batch) in enumerate(t):

        img_batch = img_batch.cuda().float()

        y_batch = y_batch.cuda()

        

        optimizer.zero_grad()

        

        rand = np.random.rand()

        if rand < 0.5:

            images, targets = mixup(img_batch, y_batch, 0.4)

            output1 = model(images)

            loss = mixup_criterion(output1, targets)

        else:

            output1 = model(img_batch)

            loss = criterion1(output1, y_batch)



        total_loss += loss.data.cpu().numpy()

        t.set_description(f'Epoch {epoch+1}/{n_epochs}, LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(batch_idx+1)))



        if history is not None:

          history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()

          history.loc[epoch + batch_idx / len(train_loader), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']

        

        loss.backward()

        optimizer.step()

        if scheduler is not None:

          scheduler.step()



def evaluate_model(epoch, scheduler=None, history=None):

    model.eval()

    loss = 0

    

    preds_1 = []

    tars_1 = []

    with torch.no_grad():

        t = tqdm(val_loader)

        for img_batch, y_batch in t:

            img_batch = img_batch.cuda().float()

            y_batch = y_batch.cuda()



            o1 = model(img_batch)



            l1 = criterion1(o1, y_batch)

            loss += l1



            for j in range(len(o1)):

              preds_1.append(torch.argmax(F.softmax(o1[j]), -1))

            for i in y_batch:

              tars_1.append(i.data.cpu().numpy())

    

    preds_1 = [p.data.cpu().numpy() for p in preds_1]

    preds_1 = np.array(preds_1).T.reshape(-1)



    acc = sklearn.metrics.recall_score(tars_1, preds_1, average='macro')

    final_score = quadratic_weighted_kappa(tars_1, preds_1)

    

    loss /= len(val_loader)

    

    if history is not None:

      history.loc[epoch, 'val_loss'] = loss.cpu().numpy()

      history.loc[epoch, 'acc'] = acc

      history.loc[epoch, 'qwk'] = final_score

    

    if scheduler is not None:

      scheduler.step(final_score)



    print(f'Dev loss: %.4f, QWK: {final_score}, Acc: {acc}'%(loss))

    

    return loss, final_score
import albumentations as A



train_transform = A.Compose([

                             A.CoarseDropout(max_holes=4, max_height=64, max_width=64, p=0.9),

                             A.OneOf([

                              A.ShiftScaleRotate(scale_limit=.15, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),

                              A.IAAAffine(shear=20, mode='constant', p=1.0),

                              A.IAAPerspective(p=1.0),

                             ], p=.9),

                             A.HorizontalFlip(p=0.5),

                             A.VerticalFlip(p=0.5),

])



fold = 0

folds = [0,1,2,3,4]

train_dataset = ImageDataset(train_df2, config.IMAGE_PATH, folds=[i for i in folds if i != fold], transform=train_transform)
nrow, ncol = 3, 6

fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))

axes = axes.flatten()

for i, ax in enumerate(axes):

    image, label = train_dataset[i]

    ax.imshow(image[0])

    ax.set_title(f'label: {label}')

plt.tight_layout()
import gc



folds = [0,1,2,3,4]



validations = []



for fold in range(5):

    print(f'Train Fold {fold+1}')

    

    history = pd.DataFrame()

    history2 = pd.DataFrame()



    torch.cuda.empty_cache()

    gc.collect()



    best = 0

    best2 = 1e10

    n_epochs = config.epochs

    

    train_dataset = ImageDataset(train_df2, config.IMAGE_PATH, folds=[i for i in folds if i != fold], transform=train_transform)

    val_dataset = ImageDataset(train_df2, config.IMAGE_PATH, folds=[fold], transform=None)

    

    BATCH_SIZE = 16

    

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    

    model = create_model()

    model = model.cuda()



    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)



    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='max', factor=0.75, verbose=True, min_lr=1e-5)



    for epoch in range(n_epochs):

        torch.cuda.empty_cache()

        gc.collect()



        train_model(epoch, optimizer, scheduler=None, history=history)



        loss, kaggle = evaluate_model(epoch, scheduler=scheduler, history=history2)



        if kaggle > best:

          best = kaggle

          print(f'Saving best model... (qwk)')

          torch.save(model.state_dict(), f'model-fld{fold+1}.pth')

        

    print()

    validations.append(best)
validations = np.array(validations)

for i,val in enumerate(validations):

    print(f'Fold {i+1}: {val}')

print(f'5fold CV: {np.mean(validations)}')
import os

import sys

import numpy as np

import pandas as pd

import cv2

from tqdm import tqdm

from timeit import default_timer as timer

import skimage.io



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torch.utils.data.sampler import *



if True:

    DATA_DIR = '/kaggle/input/prostate-cancer-grade-assessment/'

    SUBMISSION_CSV_FILE = 'submission.csv'



import warnings

warnings.filterwarnings('ignore')



# Use this to test inference

train = pd.read_csv(f'{DATA_DIR}train.csv')[:1000]

# submission = train



submission = pd.read_csv(f'{DATA_DIR}sample_submission.csv')



WIDTH = 512

HEIGHT = 512



#### net #########################################################################



def do_predict(net, inputs):

    def logit_to_probability(logit):

        probability=[]

        for l in logit:

            p = F.softmax(l)

            probability.append(p)

        return probability

    

    num_ensemble = len(net)

    for i in range(num_ensemble):

        net[i].eval()



    probability=[0,0,0,0]

    for i in range(num_ensemble):

        logit = net[i](inputs)

        prob = logit_to_probability(logit)

        probability = [p+q for p,q in zip(probability,prob)]

    

    #----

    probability = [p/num_ensemble for p in probability]

    predict = [torch.argmax(p,-1) for p in probability]

    predict = [p.data.cpu().numpy() for p in predict]

    predict = np.array(predict).T

    predict = predict.reshape(-1)



    return predict



## load net -----------------------------------

net = []



model = create_model()

model = model.cuda()

state = torch.load('model-fld1.pth') # .

model.load_state_dict(state)

net.append(model)



model = create_model()

model = model.cuda()

state = torch.load('model-fld2.pth') # .

model.load_state_dict(state)

net.append(model)



model = create_model()

model = model.cuda()

state = torch.load('model-fld3.pth') # .

model.load_state_dict(state)

net.append(model)



model = create_model()

model = model.cuda()

state = torch.load('model-fld4.pth') # .

model.load_state_dict(state)

net.append(model)



model = create_model()

model = model.cuda()

state = torch.load('model-fld5.pth') # .

model.load_state_dict(state)

net.append(model)



#------------------------------------------



from torch.utils.data import Dataset, DataLoader

from PIL import Image



class ImageDataset(Dataset):

    def __init__(self, dataframe, root_dir, transform=None):

        self.df = dataframe

        self.root_dir = root_dir

        self.transform = transform



        self.paths = self.df.image_id.values



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        img_name = self.paths[idx]

        file_path = f'{self.root_dir}{img_name}.tiff'

        

        image = skimage.io.MultiImage(file_path)

        image = cv2.resize(image[-1], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = (255 - image).astype(np.float32) / 255.

        

        if self.transform is not None:

          image = self.transform(image=image)['image']

        

        image = np.rollaxis(image, -1, 0)

        

        return image

#---------------------------------------------



def run_make_submission_csv():

    target=[]

    batch_size= 4



    if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    # Use below lines to test inference

#     if True:

#         test_dataset = ImageDataset(train, f'{DATA_DIR}train_images/', None)

        test_dataset = ImageDataset(submission, f'{DATA_DIR}test_images/', None)

        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        

        t = tqdm(test_loader)

        with torch.no_grad():

            for b, image_batch in enumerate(t):

                image_batch = image_batch.cuda().float()

                predict = do_predict(net, image_batch)

                target.append(predict)

        print('')

    #---------

    else:

        target = [[1],[1],[1]]

    target = np.concatenate(target)



    submission['isup_grade'] = target

    submission['isup_grade'] = submission['isup_grade'].astype(int)

    submission.to_csv(SUBMISSION_CSV_FILE, index=False)

    print(submission.head())



if __name__ == '__main__':

    run_make_submission_csv()



    print('\nsucess!')