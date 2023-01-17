import numpy as np

import pandas as pd

import librosa

import matplotlib.pyplot as plt

import seaborn as sns

import tqdm

import random

import os



from torch.optim import Adam



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold

import math

from collections import OrderedDict



from PIL import Image

import albumentations

from pydub import AudioSegment



import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score

device = torch.device('cuda')



import warnings

warnings.filterwarnings('ignore')
SEP = 32



MU = 256

SIGNAL_LENGTH = 160000//SEP

FOLD = 0

N_FOLDS = 5

SEED = 416

EPOCHS = 3

NPZ_DIR =  "../input/birdcall-dataset-for-wavenet/train_npz"
train = pd.read_csv("../input/birdsong-recognition/train.csv")



train = train[train["filename"].map(lambda x: x not in ["XC195038.mp3"])]

train = train.reset_index(drop=True)



# label encoding for target values

train["ebird_label"] = LabelEncoder().fit_transform(train["ebird_code"].values)
def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True 

    torch.backends.cudnn.benchmark = True 
class BirdDataset:

    def __init__(self, df, is_train):

        

        self.filename = df.filename.values

        self.ebird_label = df.ebird_label.values

        self.ebird_code = df.ebird_code.values

        self.is_train = is_train



    def __len__(self):

        return len(self.filename)

    

    def __getitem__(self, item):

        

        filename = self.filename[item].split(".")[0]

        ebird_code = self.ebird_code[item]

        ebird_label = self.ebird_label[item]



        quantized = np.load(f"{NPZ_DIR}/{ebird_code}/{filename}.wav.npz.npy").astype(int)

        

        if SIGNAL_LENGTH > len(quantized):

            onehot =  torch.eye(MU)[quantized]

            onehot = torch.cat([onehot, torch.zeros((SIGNAL_LENGTH - len(quantized), MU))], dim=0)

        elif self.is_train:

            head_i = random.sample(range(len(quantized)-SIGNAL_LENGTH), 1)[0]

            signal = quantized[head_i:head_i+SIGNAL_LENGTH]

            onehot = torch.eye(MU)[signal]

        else:

            signal = quantized[:SIGNAL_LENGTH]

            onehot = torch.eye(MU)[signal]

        

        target = ebird_label

        

        return {

            "signal" : torch.tensor(onehot, dtype=torch.float), 

            "target" : torch.tensor(target, dtype=torch.long)

        }
# ref: https://www.kaggle.com/cswwp347724/wavenet-pytorch



class Wave_Block(nn.Module):



    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):

        super(Wave_Block, self).__init__()

        self.num_rates = dilation_rates

        self.convs = nn.ModuleList()

        self.filter_convs = nn.ModuleList()

        self.gate_convs = nn.ModuleList()



        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))

        dilation_rates = [2 ** i for i in range(dilation_rates)]

        for dilation_rate in dilation_rates:

            self.filter_convs.append(

                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))

            self.gate_convs.append(

                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))

            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))



    def forward(self, x):

        x = self.convs[0](x)

        res = x

        for i in range(self.num_rates):

            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))

            x = self.convs[i + 1](x)

            res = res + x

        return res

# detail 

class Classifier(nn.Module):

    def __init__(self, inch=256, kernel_size=3):

        super().__init__()

        self.wave_block1 = Wave_Block(inch, 16, 12, kernel_size)

        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)

        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)

        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)

        self.fc = nn.Linear(128, 1)

        self.cls = nn.Linear(SIGNAL_LENGTH, 264)



    def forward(self, x):

        x = x.permute(0, 2, 1)



        x = self.wave_block1(x)

        x = self.wave_block2(x)

        x = self.wave_block3(x)



        x = self.wave_block4(x)

        x = x.permute(0, 2, 1)

        x = self.fc(x)

        x = self.cls(x.squeeze(-1))

        return x
_t = train["ebird_label"].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

trn_idx, val_idx = [(trn_idx, val_idx) for trn_idx, val_idx in skf.split(_t, _t)][FOLD]
train = train.sample(frac=1)

trn_df = train.iloc[trn_idx].reset_index(drop=True)

val_df = train.iloc[val_idx].reset_index(drop=True)
train_dataset = BirdDataset(trn_df, True)

train_data_loader = torch.utils.data.DataLoader(

        dataset = train_dataset,

        batch_size = 4*SEP,

        shuffle = True,

        pin_memory = True,

        drop_last = True

)



valid_dataset = BirdDataset(val_df, False)

valid_data_loader = torch.utils.data.DataLoader(

        dataset = valid_dataset,

        batch_size = 4*SEP,

        shuffle = False,

        pin_memory = True,

        drop_last = False

)
seed_everything(SEED)



model = Classifier()

model.to(device)

optimizer = Adam(model.parameters(), lr=0.0001)



scores, losses = [], []

best_score = 0

for epoch in range(EPOCHS):

    print(f"*** {epoch} Epoch ***")

    model.train()

    t = tqdm.notebook.tqdm(train_data_loader)

    for d in t:

        pred = model(d["signal"].to(device))

        loss = nn.CrossEntropyLoss()(pred, d["target"].to(device))

    

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    

        t.set_description(f"Train Loss = {loss.item()}")

        losses.append(loss.item())

    

    model.eval()

    f1_lst = []

    for d in valid_data_loader:

        with torch.no_grad():

            pred = model(d["signal"].to(device))

            f1 = f1_score(d["target"], pred.argmax(1).cpu(), average="micro")

            f1_lst.append(f1)

    score = sum(f1_lst)/len(f1_lst)

    print("valid f1 =", score)

    scores.append(score)

    

    if best_score < score:

        best_score = score

        torch.save(model.state_dict(), f"birdcall_wavenet_f{FOLD}_best.bin")
plt.plot(scores)
print(f"best score: {best_score}")

plt.plot(losses)