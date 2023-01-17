'''

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

!export XLA_USE_BF16=1

'''
!pip install -q --upgrade efficientnet-pytorch

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null

!python pytorch-xla-env-setup.py --version 20200420 --apt-packages libomp5 libopenblas-dev > /dev/null
import os



import torch_xla

import torch_xla.debug.metrics as met

import torch_xla.distributed.data_parallel as dp

import torch_xla.distributed.parallel_loader as pl

import torch_xla.utils.utils as xu

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.test.test_utils as test_utils









import torch



import warnings



import pandas as pd

import numpy as np

import torch.nn as nn



from sklearn.model_selection import train_test_split



from sklearn import metrics

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup



import time

import torchvision

import torch.nn as nn

from tqdm import tqdm_notebook as tqdm



from PIL import Image, ImageFile

from torch.utils.data import Dataset

import torch

import torch.optim as optim

from torchvision import transforms

from torch.optim import lr_scheduler





import sys



import gc

import os

import random



import skimage.io

import cv2

from PIL import Image

import numpy as np

import pandas as pd

import scipy as sp



import sklearn.metrics

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold



from functools import partial



from torch.utils.data import DataLoader, Dataset

import torchvision.models as models



from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip

from albumentations.pytorch import ToTensorV2





from contextlib import contextmanager

from pathlib import Path

from collections import defaultdict, Counter



warnings.filterwarnings("ignore")
#from : https://www.kaggle.com/yasufuminakama/panda-se-resnext50-classification-baseline/data

# ====================================================

# Utils

# ====================================================



@contextmanager

def timer(name):

    t0 = time.time()

    LOGGER.info(f'[{name}] start')

    yield

    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')



    

def init_logger(log_file='train.log'):

    from logging import getLogger, DEBUG, FileHandler,  Formatter,  StreamHandler

    

    log_format = '%(asctime)s %(levelname)s %(message)s'

    

    stream_handler = StreamHandler()

    stream_handler.setLevel(DEBUG)

    stream_handler.setFormatter(Formatter(log_format))

    

    file_handler = FileHandler(log_file)

    file_handler.setFormatter(Formatter(log_format))

    

    logger = getLogger('alaska2')

    logger.setLevel(DEBUG)

    logger.addHandler(stream_handler)

    logger.addHandler(file_handler)

    

    return logger



LOG_FILE = 'train.log'

LOGGER = init_logger(LOG_FILE)





def seed_torch(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_torch(seed=42)
BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"

train_imageids = pd.Series(os.listdir(BASE_PATH + '/Cover')).sort_values(ascending=True).reset_index(drop=True)

test_imageids = pd.Series(os.listdir(BASE_PATH + '/Test')).sort_values(ascending=True).reset_index(drop=True)

sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus

def append_path(pre):

    return np.vectorize(lambda file: os.path.join(BASE_PATH, pre, file))
train_filenames = np.array(os.listdir("/kaggle/input/alaska2-image-steganalysis/Cover/"))

len(train_filenames)
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus

np.random.seed(0)

positives = train_filenames.copy()

negatives = train_filenames.copy()

np.random.shuffle(positives)

np.random.shuffle(negatives)



jmipod = append_path('JMiPOD')(positives[:10000])

juniward = append_path('JUNIWARD')(positives[10000:20000])

uerd = append_path('UERD')(positives[20000:30000])



pos_paths = np.concatenate([jmipod, juniward, uerd])
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus

test_paths = append_path('Test')(sub.Id.values)

neg_paths = append_path('Cover')(negatives[:30000])
train_paths = np.concatenate([pos_paths, neg_paths])

train_labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus

train_paths, valid_paths, train_labels, valid_labels = train_test_split(

    train_paths, train_labels, test_size=0.005, random_state=2020)
len(valid_labels)
l=np.array([train_paths,train_labels])

traindataset = pd.DataFrame({ 'images': list(train_paths), 'label': train_labels},columns=['images','label'])

val_l=np.array([valid_paths,valid_labels])

validdataset = pd.DataFrame({ 'images': list(valid_paths), 'label': valid_labels},columns=['images','label'])

validdataset.head(3)
len(traindataset)
traindataset.head(2)
#i use this line of code for debugging

'''traindataset = traindataset.head(5000)

validdataset = validdataset.head(200) '''

len(traindataset)
len(validdataset)
image = Image.open(train_paths[50] )

image
class train_images(Dataset):



    def __init__(self, csv_file):



        self.data = csv_file



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        #print(idx)

        img_name =  self.data.loc[idx][0]

        image = Image.open(img_name)

        image = image.resize((512, 512), resample=Image.BILINEAR)

        label = self.data.loc[idx][1] #torch.tensor(self.data.loc[idx, 'label'])

        return {'image': transforms.ToTensor()(image),

                'label': label

                }
train_dataset = train_images(traindataset)

valid_dataset = train_images(validdataset)
!pip install efficientnet_pytorch

from efficientnet_pytorch import EfficientNet

criterion = torch.nn.BCEWithLogitsLoss() # 

num_epochs = 10

NUM_EPOCH = num_epochs

from torch.optim.lr_scheduler import OneCycleLR



BATCH_SIZE = 12

#model = torchvision.models.resnext50_32x4d(pretrained=True)

#model.load_state_dict(torch.load("../input/pytorch-pretrained-models/resnet101-5d3b4d8f.pth"))

model = EfficientNet.from_name('efficientnet-b3')



#model.avg_pool = nn.AdaptiveAvgPool2d(1)

num_ftrs = model._fc.in_features

model._fc = nn.Linear(num_ftrs, 1)

#model.load_state_dict(torch.load("../input/pytorch-transfer-learning-baseline/model.bin"))

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#model
# https://www.kaggle.com/anokas/weighted-auc-metric-updated



def alaska_weighted_auc(y_true, y_valid):

    tpr_thresholds = [0.0, 0.4, 1.0]

    weights = [2,   1]



    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)



    # size of subsets

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])



    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

    normalization = np.dot(areas, weights)



    competition_metric = 0

    for idx, weight in enumerate(weights):

        y_min = tpr_thresholds[idx]

        y_max = tpr_thresholds[idx + 1]

        mask = (y_min < tpr) & (tpr < y_max)

        # pdb.set_trace()



        x_padding = np.linspace(fpr[mask][-1], 1, 100)



        x = np.concatenate([fpr[mask], x_padding])

        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

        y = y - y_min  # normalize such that curve starts at y=0

        score = metrics.auc(x, y)

        submetric = score * weight

        best_subscore = (y_max - y_min) * weight

        competition_metric += submetric



    return competition_metric / normalization
#https://www.kaggle.com/dhananjay3/pytorch-xla-for-tpu-with-multiprocessing

#https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59/data



def train_model():

    global train_dataset, valid_dataset

    

    torch.manual_seed(42)

    

    train_sampler = torch.utils.data.distributed.DistributedSampler(

        train_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True)

    

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=train_sampler,

        num_workers=0,

        drop_last=True) # print(len(train_loader))

    

    valid_sampler = torch.utils.data.distributed.DistributedSampler(

        valid_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        )

        

    valid_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=BATCH_SIZE ,

        sampler=valid_sampler,

        shuffle=False,

        num_workers=0,

        drop_last=True)

    

    #xm.master_print(f"Train for {len(train_loader)} steps per epoch")

    LOGGER.debug(f"Train for {len(train_loader)} steps per epoch")

    # Scale learning rate to num cores

    lr  = 0.001 * xm.xrt_world_size()



    # Get loss function, optimizer, and model

    device = xm.xla_device()



    #model = model()

    '''

    for param in model.base_model.parameters(): # freeze some layers

        param.requires_grad = False'''

    

    

    global model

    

    model = model.to(device)



    criterion = torch.nn.MSELoss() #  BCEWithLogitsLoss

    #criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10)

    

    



    

    def train_loop_fn(loader):

        tracker = xm.RateTracker()

        model.train()

        #xm.master_print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        LOGGER.debug('Epoch {}/{}'.format(epoch, num_epochs - 1))

        #xm.master_print('-' * 10)



        LOGGER.debug('-' * 10)

        scheduler.step()

        

        running_loss = 0.0

        tk0 = tqdm(loader, total=int(len(train_loader)))

        counter = 0

        for bi, d in enumerate(tk0):

            inputs = d["image"]

            labels = d["label"].view(-1, 1)

            inputs = inputs.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.float)

            #labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            #with torch.set_grad_enabled(True):

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            #loss = criterion(outputs, torch.max(labels, 1)[1])

            loss.backward()

            xm.optimizer_step(optimizer)

            running_loss += loss.item() * inputs.size(0)

            #print(running_loss)

            counter += 1

            tk0.set_postfix(loss=(running_loss / (counter * BATCH_SIZE)))

        epoch_loss = running_loss / len(train_loader)

        

        #xm.master_print('Training Loss: {:.8f}'.format(epoch_loss))

        LOGGER.debug('Training Loss: {:.8f}'.format(epoch_loss))



                

    def test_loop_fn(loader):

        

        tk0 = tqdm(loader, total=int(len(valid_loader)))

        counter = 0

        total_samples, correct = 0, 0

        for bi, d in enumerate(tk0):

            inputs = d["image"]

            labels = d["label"].view(-1, 1)

            inputs = inputs.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.float)

            #labels = labels.to(device, dtype=torch.long)

            #optimizer.zero_grad()

            

            #with torch.no_grad():

                

            output = model(inputs)

                

            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(labels.view_as(pred)).sum().item()

            total_samples += inputs.size()[0]

        accuracy = 100.0 * correct / total_samples

        

        

        #auc_score = alaska_weighted_auc(labels.cpu().numpy(), output.cpu().numpy())

        #LOGGER.debug("auc_score according to competition metric = {} ".format(auc_score))

        #print('[xla:{}] Accuracy={:.4f}%'.format(xm.get_ordinal(), accuracy), flush=True)

        model.train()

        return accuracy



    # Train - valid  loop

    accuracy = []

    for epoch in range(1, num_epochs + 1):

        start = time.time()

        para_loader = pl.ParallelLoader(train_loader, [device])

        train_loop_fn(para_loader.per_device_loader(device))

        

        para_loader = pl.ParallelLoader(valid_loader, [device])

        accuracy.append(test_loop_fn(para_loader.per_device_loader(device)))

        #xm.master_print("Finished training epoch {}  Val-Acc {:.4f} in {:.4f} sec".format(epoch, accuracy[-1],   time.time() - start))        

        

        

        LOGGER.debug("Finished training epoch {}  Val-Acc {:.4f} in {:.4f} sec".format(epoch, accuracy[-1],   time.time() - start))   

        valauc = accuracy[-1]

        if(epoch>9):

            xm.save(model.state_dict(), f"./epoch{epoch}valauc{valauc}.bin")

    return accuracy
# Start training processes



def _mp_fn(rank, flags):

    global acc_list

    torch.set_default_tensor_type('torch.FloatTensor')

    res = train_model()



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
xm.get_xla_supported_devices()
class test_images(Dataset):



    def __init__(self, csv_file):



        self.data = csv_file



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        img_name =  self.data.loc[idx][0]

        image = Image.open(img_name)

        image = image.resize((512, 512), resample=Image.BILINEAR)

        #label = self.data.loc[idx][1] #torch.tensor(self.data.loc[idx, 'label'])

        #image = self.transform(image)

        return {'image': transforms.ToTensor()(image)}



testdataset = pd.DataFrame({ 'images': list(test_paths)},columns=['images'])

#testdataset.head(2)

testdataset = test_images(testdataset)
Test_BATCH_SIZE = 1    

test_sampler = torch.utils.data.distributed.DistributedSampler(

      testdataset,

      num_replicas=xm.xrt_world_size(),

      rank=xm.get_ordinal()

      )



test_data_loader = torch.utils.data.DataLoader(

    testdataset,

    batch_size=Test_BATCH_SIZE,

    #sampler=test_sampler,

    #drop_last=True,

    num_workers=0

)



device = xm.xla_device()



model = model.to(device)



testpara_loader = pl.ParallelLoader(test_data_loader, [device])



sub["Label"] = pd.to_numeric(sub["Label"].astype(float))

%%time

#test_loader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False) # test_set contains only images directory





test_loader =  testpara_loader.per_device_loader(device)





#print(type(test_loader))

'''

for param in model.parameters():

    param.requires_grad = False '''



prediction_list = []

tk0 = tqdm(test_loader)



for i, x_batch in enumerate(tk0):

    

    x_batch = x_batch["image"].to(device)

    

    #x_batch.to(device)

    #pred =  model(x_batch)

    pred =  model(x_batch.to(device))

    #prediction_list.append(pred.cpu())

    #print( type(pred.item()))

    #print("\n")

    sub.Label[i] = pred.item()

    #print(sub.Label[i])
sub.to_csv('submission.csv', index=False)

sub.head(5)