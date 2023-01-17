# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 



# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import IPython.display as ipd



import torch

import random

import numpy as np

import pandas as pd

import wave

from scipy.io import wavfile

import os

import librosa

from librosa.feature import melspectrogram

import warnings

from sklearn.utils import shuffle

from sklearn.utils import class_weight

from PIL import Image

from uuid import uuid4

import sklearn

from tqdm import tqdm

import soundfile as sf

from sklearn.metrics import f1_score



import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset,DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

from torch.nn import functional as F

from glob import glob

import sklearn

from torch import nn

import warnings

import cv2



warnings.filterwarnings("ignore") 

warnings.filterwarnings("ignore", category=DeprecationWarning) 



SEED = 42



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)

import pandas as pd



IM_SIZE=256
PATHS = dict()

for x in os.listdir('/kaggle/input/'):

        for y in os.listdir('/kaggle/input/'+x):

            if 'csv' not in y:

                for z in os.listdir('/kaggle/input/'+x+'/'+y):

                        PATHS[z.split('.')[0]]='/kaggle/input/'+x+'/'+y+'/'+z
train = pd.read_csv('/kaggle/input/birdsong-resampled-train-audio-04/train_mod.csv')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



train["fold"] = -1

for fold_id, (train_index, val_index) in enumerate(skf.split(train, train["ebird_code"])):

    train.iloc[val_index, -1] = fold_id

    

train.fold.value_counts()
sf.read(PATHS['XC124527'])
PATHS[train.loc[0].filename[:-4]]
species=train.species.value_counts().keys()

BIRD_NUM=dict(zip(species,range(len(species))))

NUM_BIRD=dict(zip(range(len(species)),species))
PATH='/kaggle/input/birdsong-recognition/train_audio/'

PERIOD=5

melspectrogram_parameters= {'n_mels': 128, 'fmin': 20, 'fmax': 16000}







def mono_to_color(

    X: np.ndarray, mean=None, std=None,

    norm_max=None, norm_min=None, eps=1e-6

):

    # Stack X as [X,X,X]

    X = np.stack([X, X, X], axis=-1)



    # Standardize

    mean = mean or X.mean()

    X = X - mean

    std = std or X.std()

    Xstd = X / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min

    if (_max - _min) > eps:

        # Normalize to [0, 255]

        V = Xstd

        V[V < norm_min] = norm_min

        V[V > norm_max] = norm_max

        V = 255 * (V - norm_min) / (norm_max - norm_min)

        V = V.astype(np.uint8)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return V
class DatasetRetriever(Dataset):

    def __init__(self,

                 df,

                 size=None,

                 train=True,

                 melspectrogram_parameters=None

                ):

        self.img_size=128

        self.melspectrogram_parameters = melspectrogram_parameters

        self.df = df.reset_index(drop=True)

        self.train = train

        

        self.change_size(size)

    def change_size(self,size):

        IM_SIZE=size

        if self.train:

            self.transform=None

        else:

            self.transform =None

    def get_spectogram(self,item):



        #wav_path = PATH+item.ebird_code+'/'+item.filename

        wave_data, sr = sf.read(PATHS[item.filename.split('.')[0]])#librosa.load(wav_path)

        y, _ = librosa.effects.trim(wave_data)

        sample_length = 5*sr

        len_y = len(y)

        effective_length = sr * PERIOD

        if len_y < effective_length:

                    new_y = np.zeros(effective_length, dtype=y.dtype)

                    start = np.random.randint(effective_length - len_y) if self.train else 0

                    new_y[start:start + len_y] = y

                    y = new_y.astype(np.float32)

        elif len_y > effective_length:

                    start = np.random.randint(len_y - effective_length) if self.train else 0

                    y = y[start:start + effective_length].astype(np.float32) 

        else:

                    y = y.astype(np.float32)

        

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)

       

        melspec = librosa.power_to_db(melspec).astype(np.float32)

      

        

        '''if self.spectrogram_transforms:

            melspec = self.spectrogram_transforms(melspec)

        else:

            pass'''

        

        image = mono_to_color(melspec)

        height, width, _ = image.shape

        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))

        image = np.moveaxis(image, 2, 0)

        image = (image / 255.0).astype(np.float32)



        #  labels = np.zeros(len(BIRD_CODE), dtype="i")

        labels = np.zeros(len(BIRD_NUM), dtype="f")

        labels[BIRD_NUM[item.species]] = 1

       

        return image, labels

        

    

    def __len__(self):

        return self.df.shape[0]



    def __getitem__(self, index):

        item = self.df.iloc[index]

    

        return self.prepare_img(item)        

    def get_labels(self):

        return list(self.df.target.values)

    def prepare_img(self,item):

        image,target =self.get_spectogram(item)

        return torch.tensor(image), torch.tensor(target)
ds=DatasetRetriever(train.reset_index(drop=True),

                 size=None,

                 train=True,

                 melspectrogram_parameters=melspectrogram_parameters)
x,y=ds[0]
plt.imshow(x.permute(1,2,0))
df=train

fold_number=0

train_dataset = DatasetRetriever(

    df=df[df.fold != fold_number],

    size=IM_SIZE,

    train=True,melspectrogram_parameters=melspectrogram_parameters

)





validation_dataset = DatasetRetriever(

    df=df[df.fold==fold_number],

    size=IM_SIZE,

    train=False,

    melspectrogram_parameters=melspectrogram_parameters)
plt.imshow(train_dataset[0][0].permute(1,2,0))
import numpy as np



class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):

        self.reset()

        self.threshold=0.5

    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0

        

        self.y_true = np.zeros((1,264))#np.array([1])

        self.y_pred = np.zeros((1,264))#np.array([1])

    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count

class RocAucMeter(object):

    def __init__(self):

        self.reset()



    def reset(self):

        self.y_true = np.zeros((1,264))

        self.y_pred = np.zeros((1,264))

        self.score = []



    def update(self, y_true, y_pred):

        

        #print('in')

        y_true = y_true.cpu().numpy()

        y_pred =y_pred.detach().cpu().numpy()

        self.y_true = np.concatenate((self.y_true, y_true))

        self.y_pred = np.concatenate((self.y_pred, y_pred))

        self.score.append((y_true.argmax(axis=1)==y_pred.argmax(axis=1)).mean())#$, average='samples')

        #print(self.score,'<<<<')

    @property

    def avg(self):

        return np.mean(self.score)

    

class APScoreMeter(object):

    def __init__(self):

        self.reset()

    def reset(self):

        self.y_true = np.zeros((1,264))

        self.y_pred = np.zeros((1,264))

        self.score = []

    def update(self, y_true, y_pred):

        y_true = y_true.cpu().numpy()

        y_pred = np.where(y_pred.sigmoid().detach().cpu().numpy() > 0.5, 1, 0)

        self.y_true = np.concatenate((self.y_true, y_true))

        self.y_pred = np.concatenate((self.y_pred, y_pred))

        self.score.append(f1_score(y_true, y_pred, average='samples'))

    @property

    def avg(self):

        return np.mean(self.score)
class Fitter:

    

    def __init__(self, model, device, config, folder):

        self.config = config

        self.epoch = 0



        self.base_dir = f'./{folder}'

        if not os.path.exists(self.base_dir):

            os.makedirs(self.base_dir)



        self.log_path = f'{self.base_dir}/log.txt'

        self.best_score = 0

        self.best_loss = 10**5

        self.best_ap = 0

        

        self.model = model

        self.device = device



        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [

            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

        ] 



        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)



#         self.criterion = FocalLoss(logits=True).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

        self.log(f'Fitter prepared. Device is {self.device}')



    def fit(self, train_loader, validation_loader):

        for e in range(self.config.n_epochs):

            if self.config.verbose:

                lr = self.optimizer.param_groups[0]['lr']

                timestamp = datetime.utcnow().isoformat()

                self.log(f'\n{timestamp}\nLR: {lr}')



            t = time.time()

            summary_loss, roc_auc_scores, ap_scores = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, roc_auc: {roc_auc_scores.avg:.5f}, ap: {ap_scores.avg:.5f}, time: {(time.time() - t):.5f}')



            t = time.time()

            summary_loss, roc_auc_scores, ap_scores = self.validation(validation_loader)



            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, roc_auc: {roc_auc_scores.avg:.5f}, ap: {ap_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            if summary_loss.avg < self.best_loss:

                self.best_loss = summary_loss.avg

                self.save_model(f'{self.base_dir}/best-loss-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

                for path in sorted(glob(f'{self.base_dir}/best-loss-checkpoint-*epoch.bin'))[:-2]:

                    os.remove(path)

                    

            if roc_auc_scores.avg > self.best_score:

                self.best_score = roc_auc_scores.avg

                self.save_model(f'{self.base_dir}/best-score-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

                for path in sorted(glob(f'{self.base_dir}/best-score-checkpoint-*epoch.bin'))[:-2]:

                    os.remove(path)

                    

            if ap_scores.avg > self.best_ap:

                self.best_ap = ap_scores.avg

                self.save_model(f'{self.base_dir}/best-ap-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

                for path in sorted(glob(f'{self.base_dir}/best-ap-checkpoint-*epoch.bin'))[:-2]:

                    os.remove(path)



            if self.config.validation_scheduler:

                self.scheduler.step(metrics=summary_loss.avg)



            self.epoch += 1



    def validation(self, val_loader):

        self.model.eval()

        summary_loss = AverageMeter()

        roc_auc_scores = RocAucMeter()

        ap_scores = APScoreMeter()

        t = time.time()

        for step, (images, targets) in enumerate(val_loader):

            if self.config.verbose:

                if step % self.config.verbose_step == 0:

                    print(

                        f'Val Step {step}/{len(val_loader)}, ' + \

                        f'summary_loss: {summary_loss.avg:.5f}, roc_auc: {roc_auc_scores.avg:.5f}, ap: {ap_scores.avg:.5f} ' + \

                        f'time: {(time.time() - t):.5f}', end='\r'

                    )

            with torch.no_grad():

                targets = targets.to(self.device).float()

                batch_size = images.shape[0]

                images = images.to(self.device).float()

                outputs = self.model(images)

                loss = self.criterion(outputs, targets)

                roc_auc_scores.update(targets, outputs)

                ap_scores.update(targets, outputs)

                summary_loss.update(loss.detach().item(), batch_size)



        return summary_loss, roc_auc_scores, ap_scores



    def train_one_epoch(self, train_loader):

        self.model.train()

        summary_loss = AverageMeter()

        roc_auc_scores = RocAucMeter()

        ap_scores = APScoreMeter()

        t = time.time()

        for step, (images, targets) in enumerate(train_loader):

            print(

                        f'Train Step {step}/{len(train_loader)}, ' + \

                        f'summary_loss: {summary_loss.avg:.5f}, roc_auc: {roc_auc_scores.avg:.5f}, ap: {ap_scores.avg:.5f} ' + \

                        f'time: {(time.time() - t):.5f}', end='\r'

                    )

            if self.config.verbose:

                if step % self.config.verbose_step == 0:

                    print(

                        f'Train Step {step}/{len(train_loader)}, ' + \

                        f'summary_loss: {summary_loss.avg:.5f}, roc_auc: {roc_auc_scores.avg:.5f}, ap: {ap_scores.avg:.5f} ' + \

                        f'time: {(time.time() - t):.5f}', end='\r'

                    )

            

            targets = targets.to(self.device).float()

            images = images.to(self.device).float()

            batch_size = images.shape[0]



            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, targets)

            loss.backward()

            

            roc_auc_scores.update(targets, outputs)

            ap_scores.update(targets, outputs)

            summary_loss.update(loss.detach().item(), batch_size)



            self.optimizer.step()



            if self.config.step_scheduler:

                self.scheduler.step()



        return summary_loss, roc_auc_scores, ap_scores

    

    def save_model(self, path):

        self.model.eval()

        torch.save(self.model.state_dict(),path)



    def save(self, path):

        self.model.eval()

        torch.save({

            'model_state_dict': self.model.state_dict(),

            'optimizer_state_dict': self.optimizer.state_dict(),

            'scheduler_state_dict': self.scheduler.state_dict(),

            'best_score': self.best_score,

            'best_ap': self.best_ap,

            'best_loss': self.best_loss,

            'epoch': self.epoch,

        }, path)



    def load(self, path):

        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_score = checkpoint['best_score']

        self.best_ap = checkpoint['best_ap']

        self.best_loss = checkpoint['best_loss']

        self.epoch = checkpoint['epoch']

        

    def log(self, message):

        if self.config.verbose:

            print(message)

        with open(self.log_path, 'a+') as logger:

            logger.write(f'{message}\n')
train.species.value_counts()
import torchvision.models as models



import torch.nn as nn

import torch.nn.functional as F



class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.effnet=nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self._fc =  nn.Linear(in_features=512, out_features=264, bias=True)



    def forward(self, x):

        if np.random.rand() > 0.7 and self.training:

            ids=torch.randperm(x.size()[0])

            x_ = x[ids,:,:,:].detach()

            x=x*0.9+x_*0.1

        xx=self.avg(self.effnet(x)).view(x.size(0),-1)

        x=self._fc(xx)

        return x

def get_net():

    net = Model()

    #net._fc = nn.Linear(in_features=2048, out_features=2, bias=True)

    return net
net = get_net()

#net.load_state_dict(torch.load('/kaggle/input/bird-call/fold0/best-score-checkpoint-021epoch.bin'))
class TrainGlobalConfig:

    num_workers = 2

    batch_size = 64

    n_epochs = 30

    lr = 0.001#0.5 * 1e-5

    

    # -------------------

    verbose = True

    verbose_step = 50

    # -------------------



    # --------------------

    step_scheduler = False  # do scheduler.step after optimizer.step

    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau

    scheduler_params = dict(

        mode='max',

        factor=0.95,

        patience=0,

        verbose=False, 

        threshold=0.0001,

        threshold_mode='abs',

        cooldown=0, 

        min_lr=1e-7,

        eps=1e-08

    )

    '''

    SchedulerClass = torch.optim.lr_scheduler.CosineAnnealingLR

    scheduler_params = dict(

       T_max=10, 

    )'''

    # --------------------



    # -------------------

    criterion = nn.BCEWithLogitsLoss()

    # -------------------
from catalyst.data.sampler import BalanceClassSampler



def train_fold(fold,size):





    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=TrainGlobalConfig.batch_size,

        pin_memory=False,

        shuffle=True,

        drop_last=True,

        num_workers=TrainGlobalConfig.num_workers,

        

    )

    val_loader = torch.utils.data.DataLoader(

        validation_dataset, 

        batch_size=TrainGlobalConfig.batch_size,

        num_workers=TrainGlobalConfig.num_workers,

        shuffle=False,

        pin_memory=False,

    )



    fitter = Fitter(model=net, device=torch.device('cuda:0'), config=TrainGlobalConfig, folder=f'fold{fold}')

    fitter.fit(train_loader, val_loader)
net=net.to(torch.device('cuda'))

import time

from datetime import datetime



train_fold(fold=0,size=256)
train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=TrainGlobalConfig.batch_size,

        pin_memory=False,

        shuffle=True,

        drop_last=True,

        num_workers=TrainGlobalConfig.num_workers,

        

    )
for x,y in train_loader:

    break
y.argmax(1)
res = net(x.to(torch.device('cuda')))
(res.argmax(1).cpu()==y.argmax(1)).float().mean()