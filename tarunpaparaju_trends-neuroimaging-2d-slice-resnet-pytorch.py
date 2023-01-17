!pip install -q colored

!pip install -q torchviz
import os

import gc

import cv2

import time

import h5py

import colored

from colored import fg, bg, attr



from skimage import measure

from plotly.offline import iplot

from plotly import figure_factory as FF

from IPython.display import Markdown, display



import numpy as np

import pandas as pd

from random import randint

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt



import torch

from torchviz import make_dot

torch.backends.cudnn.benchmark = True



import torch.nn as nn

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.models import resnet18, densenet121, mobilenet_v2
EPOCHS = 2

SPLIT = 0.8

LR = (1e-4, 1e-3)

MODEL_SAVE_PATH = "resnet_model"



W = 64

H = 64

BATCH_SIZE = 32

VAL_BATCH_SIZE = 32

DATA_PATH = '../input/trends-assessment-prediction/'
TEST_MAP_PATH = DATA_PATH + 'fMRI_test/'

TRAIN_MAP_PATH = DATA_PATH + 'fMRI_train/'



FEAT_PATH = DATA_PATH + 'fnc.csv'

TARG_PATH = DATA_PATH + 'train_scores.csv'

SAMPLE_SUB_PATH = DATA_PATH + 'sample_submission.csv'



TEST_IDS = [map_id[:-4] for map_id in sorted(os.listdir(TEST_MAP_PATH))]

TRAIN_IDS = [map_id[:-4] for map_id in sorted(os.listdir(TRAIN_MAP_PATH))]
targets = pd.read_csv(TARG_PATH)

targets = targets.fillna(targets.mean())

sample_submission = pd.read_csv(SAMPLE_SUB_PATH)



features = pd.read_csv(FEAT_PATH)

test_df = features.query('Id in {}'.format(TEST_IDS)).reset_index(drop=True)

train_df = features.query('Id in {}'.format(TRAIN_IDS)).reset_index(drop=True)
targets.head()
features.head()
sample_submission.head()
def display_maps(idx):

    path = TRAIN_MAP_PATH + str(train_df.Id[idx])

    all_maps = h5py.File(path + '.mat', 'r')['SM_feature'][()]

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20, 12.5))



    plt.set_cmap('gray')

    for i in range(5):

        idx_1, idx_2, idx_3 = randint(0, 51), randint(0, 62), randint(0, 52)



        proj_1 = all_maps[:, idx_1, :, :].transpose(1, 2, 0)

        proj_2 = all_maps[:, :, idx_2, :].transpose(1, 2, 0)

        proj_3 = all_maps[:, :, :, idx_3].transpose(1, 2, 0)

        ax[0, i].imshow(cv2.resize(proj_1[:, :, 0], (H, W)))

        ax[1, i].imshow(cv2.resize(proj_2[:, :, 0], (H, W)))

        ax[2, i].imshow(cv2.resize(proj_3[:, :, 0], (H, W)))

        ax[0, i].set_title('Z-section {}'.format(i), fontsize=12)

        ax[1, i].set_title('Y-section {}'.format(i), fontsize=12)

        ax[2, i].set_title('X-section {}'.format(i), fontsize=12)



    plt.suptitle('Id: {}'.format(train_df.Id[idx])); plt.show()
display_maps(0)
display_maps(1)
display_maps(2)
class TReNDSDataset(Dataset):

    def __init__(self, data, targets, map_path, is_train):

        self.data = data

        self.is_train = is_train

        self.map_path = map_path

        self.map_id = self.data.Id

        if is_train: self.targets = targets

            

    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        path = self.map_path + str(self.map_id[idx])

        all_maps = h5py.File(path + '.mat', 'r')['SM_feature'][()]

        cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

        

        idx_1, idx_2, idx_3 = randint(0, 51), randint(0, 62), randint(0, 52)

        proj_1 = cv2.resize(all_maps[:, idx_1, :, :].transpose(1, 2, 0), (H, W))

        proj_2 = cv2.resize(all_maps[:, :, idx_2, :].transpose(1, 2, 0), (H, W))

        proj_3 = cv2.resize(all_maps[:, :, :, idx_3].transpose(1, 2, 0), (H, W))

        features = np.concatenate([proj_1, proj_2, proj_3], axis=2).transpose(2, 0, 1)

        

        if not self.is_train:

            return torch.FloatTensor(features)

        else:

            i = self.map_id[idx]

            targets = self.targets.query('Id == {}'.format(i)).values

            targets = np.repeat(targets[:, 1:], 159, 0).reshape(-1, 5)

            return torch.FloatTensor(features), torch.FloatTensor(targets)
class ResNetModel(nn.Module):

    def __init__(self):

        super(ResNetModel, self).__init__()

        

        self.identity = lambda x: x

        self.dense_out = nn.Linear(16, 5)

        self.dense_in = nn.Linear(512, 16)

        self.resnet = resnet18(pretrained=True, progress=False)

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        

    def forward(self, img):

        img = img.reshape(-1, 1, H, W)

        feat = self.resnet(img.repeat(1, 3, 1, 1))

        

        conc = self.dense_in(feat.squeeze())

        return self.identity(self.dense_out(conc))
model = ResNetModel()

x = torch.randn(1, 3, 64, 64).requires_grad_(True)

y = model(x)

make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
del model, x, y

gc.collect()
def weighted_nae(inp, targ):

    W = torch.FloatTensor([0.3, 0.175, 0.175, 0.175, 0.175])

    return torch.mean(torch.matmul(torch.abs(inp - targ), W.cuda()/torch.mean(targ, axis=0)))
def print_metric(data, batch, epoch, start, end, metric, typ):

    time = np.round(end - start, 1)

    time = "Time: %s{}%s s".format(time)



    if typ == "Train":

        pre = "BATCH %s" + str(batch-1) + "%s  "

    if typ == "Val":

        pre = "EPOCH %s" + str(epoch+1) + "%s  "

    

    fonts = (fg(216), attr('reset'))

    value = np.round(data.item(), 3)

    t = typ, metric, "%s", value, "%s"



    print(pre % fonts , end='')

    print("{} {}: {}{}{}".format(*t) % fonts + "  " + time % fonts)
val_out_shape = -1, 5

train_out_shape = -1, 5



split = int(SPLIT*len(train_df))

val = train_df[split:].reset_index(drop=True)

train = train_df[:split].reset_index(drop=True)



test_set = TReNDSDataset(test_df, None, TEST_MAP_PATH, False)

test_loader = DataLoader(test_set, batch_size=VAL_BATCH_SIZE)
def train_resnet18():

    def cuda(tensor):

        return tensor.cuda()

   

    val_set = TReNDSDataset(val, targets, TRAIN_MAP_PATH, True)

    val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE)

    train_set = TReNDSDataset(train, targets, TRAIN_MAP_PATH, True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)



    network = cuda(ResNetModel())

    optimizer = Adam([{'params': network.resnet.parameters(), 'lr': LR[0]},

                      {'params': network.dense_in.parameters(), 'lr': LR[1]},

                      {'params': network.dense_out.parameters(), 'lr': LR[1]}])



    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,

                                  patience=2, verbose=True, eps=1e-6)

    start = time.time()

    for epoch in range(EPOCHS):

        batch = 1

        fonts = (fg(216), attr('reset'))

        print(("EPOCH %s" + str(epoch+1) + "%s") % fonts)



        for train_batch in train_loader:

            train_img, train_targs = train_batch

           

            network.train()

            network = cuda(network)

            train_preds = network.forward(cuda(train_img))

            train_targs = train_targs.reshape(train_out_shape)

            train_loss = weighted_nae(train_preds, cuda(train_targs))



            optimizer.zero_grad()

            train_loss.backward()



            optimizer.step()

            end = time.time()

            batch = batch + 1

            print_metric(train_loss, batch, epoch, start, end, metric="loss", typ="Train")

            

        print("\n")

           

        network.eval()

        for val_batch in val_loader:

            img, targ = val_batch

            val_preds, val_targs = [], []



            with torch.no_grad():

                img = cuda(img)

                network = cuda(network)

                pred = network.forward(img)

                val_preds.append(pred); val_targs.append(targ)



        val_preds = torch.cat(val_preds, axis=0)

        val_targs = torch.cat(val_targs, axis=0)

        val_targs = val_targs.reshape(val_out_shape)

        val_loss = weighted_nae(val_preds, cuda(val_targs))

        

        avg_preds = []

        avg_targs = []

        for idx in range(0, len(val_preds), 159):

            avg_preds.append(val_preds[idx:idx+159].mean(axis=0))

            avg_targs.append(val_targs[idx:idx+159].mean(axis=0))

            

        avg_preds = torch.stack(avg_preds, axis=0)

        avg_targs = torch.stack(avg_targs, axis=0)

        loss = weighted_nae(avg_preds, cuda(avg_targs))

        

        end = time.time()

        scheduler.step(val_loss)

        print_metric(loss, None, epoch, start, end, metric="loss", typ="Val")

        

        print("\n")

   

    network.eval()

    if os.path.exists(TEST_MAP_PATH):



        test_preds = []

        for test_img in test_loader:

            with torch.no_grad():

                network = cuda(network)

                test_img = cuda(test_img)

                test_preds.append(network.forward(test_img))

        

        avg_preds = []

        test_preds = torch.cat(test_preds, axis=0)

        for idx in range(0, len(test_preds), 159):

            avg_preds.append(test_preds[idx:idx+159].mean(axis=0))



        torch.save(network.state_dict(), MODEL_SAVE_PATH + ".pt")

        return torch.stack(avg_preds, axis=0).detach().cpu().numpy()
print("STARTING TRAINING ...\n")



test_preds = train_resnet18()

    

print("ENDING TRAINING ...")
sample_submission.Predicted = test_preds.flatten()
sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)