#/kaggle/input/digit-recognizer/sample_submission.csv

#/kaggle/input/digit-recognizer/test.csv

#/kaggle/input/digit-recognizer/train.csv
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import torch

from torch import nn, optim

from torch.utils.data import DataLoader as DL

from torch.utils.data import Dataset

from torch.nn.utils import weight_norm as WN

import torch.nn.functional as F



from sklearn.model_selection import StratifiedKFold



from time import time

import random as r



MAX_VALUE = 255
def breaker():

    print("\n" + 30*"-" + "\n")



def head(x, no_of_ele=5):

    breaker()

    print(x[:no_of_ele])

    breaker()
tr_Set = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

ts_Set = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



breaker()

print("Train Set Shape :", repr(tr_Set.shape))

breaker()

print("Test Set Shape  :", repr(ts_Set.shape))

breaker()



X, y = tr_Set.iloc[:, 1:].copy().values, tr_Set.iloc[:, 0].copy().values

X_test = ts_Set.copy().values



del tr_Set, ts_Set
X, X_test = np.divide(X, MAX_VALUE), np.divide(X_test, MAX_VALUE)

num_features = X.shape[1]

num_obs_test = X_test.shape[0]
class DS(Dataset):

    def __init__(this, X=None, y=None, mode="train"):

        this.mode = mode

        this.X = X

        if mode == "train":

            this.y = y

    

    def __len__(this):

        return this.X.shape[0]

    

    def __getitem__(this, idx):

        if this.mode == "train":

            return torch.FloatTensor(this.X[idx]), torch.LongTensor(this.y[idx])

        else:

            return torch.FloatTensor(this.X[idx])
class ANN_CFG():

    tr_batch_size = 128

    va_batch_size = 128

    ts_batch_size = 128

    

    epochs = 25

    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    IL = num_features

    HL = [2048, 2048]

    OL = 10



cfg = ANN_CFG()
class ANN(nn.Module):

    def __init__(this, IL=None, HL=None, OL=None):

        super(ANN, this).__init__()

        

        this.DP  = nn.Dropout(p=0.25) 

        

        this.BN1 = nn.BatchNorm1d(IL)

        this.FC1 = WN(nn.Linear(IL, HL[0]))

        

        this.BN2 = nn.BatchNorm1d(HL[0])

        this.FC2 = WN(nn.Linear(HL[0], HL[1]))

        

        this.BN3 = nn.BatchNorm1d(HL[1])

        this.FC3 = WN(nn.Linear(HL[1], OL))

        

    def getOptimizer(this):

        return optim.Adam(this.parameters(), lr=1e-3, weight_decay=0)

    

    def forward(this, x):

        x = this.BN1(x)

        x = F.relu(this.FC1(x))

        

        x = this.BN2(x)

        x = F.relu(this.FC2(x))

        

        x = this.BN3(x)

        x = F.log_softmax(this.FC3(x), dim=1)

        return x
def train_fn(X=None, y=None):

    bestLoss = {"train" : np.inf, "valid" : np.inf}

    

    LP = []

    name_getter = []

    n_folds = 4

    fold = 0

    breaker()

    start_time = time()

    for tr_idx, va_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0).split(X, y):

        print("Processing Fold {fold} ...".format(fold=fold))

        

        X_train, X_valid, y_train, y_valid = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

        

        tr_data_setup = DS(X_train, y_train.reshape(-1,1))

        va_data_setup = DS(X_valid, y_valid.reshape(-1,1))

        

        dataloaders = {"train" : DL(tr_data_setup, batch_size=cfg.tr_batch_size, shuffle=True, generator=torch.manual_seed(0)),

                       "valid" : DL(va_data_setup, batch_size=cfg.va_batch_size, shuffle=False)}

        

        model = ANN(cfg.IL, cfg.HL, cfg.OL)

        model.to(cfg.device)

        

        optimizer = model.getOptimizer()

        

        for e in range(cfg.epochs):

            epochLoss = {"train" : 0, "valid" : 0}

            for phase in ["train", "valid"]:

                if phase == "train":

                    model.train()

                else:

                    model.eval()

                lossPerPass = 0

                

                for feats, label in dataloaders[phase]:

                    feats, label = feats.to(cfg.device), label.to(cfg.device).view(-1)

                    

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):

                        output = model(feats)

                        loss = nn.NLLLoss()(output, label)

                        if phase == "train":

                            loss.backward()

                            optimizer.step()

                    lossPerPass += (loss.item()/label.shape[0])

                epochLoss[phase] = lossPerPass

            LP.append(epochLoss)

            if epochLoss["valid"] < bestLoss["valid"]:

                bestLoss = epochLoss

                name = "./Model_Fold_{fold}.pt".format(fold=fold)

                name_getter.append(name)

                torch.save(model.state_dict(), name)

        fold += 1

    

    breaker()

    print("Time Taken to Train {fold} folds for {e} epochs : {:.2f} minutes".format((time()-start_time)/60, fold=fold, e=cfg.epochs))

    breaker()

    

    return LP, name_getter, model

          

def eval_fn(model=None, names=None, dataloader=None):

    final_Pred = np.zeros((num_obs_test, 1))



    for name in names:

        model.load_state_dict(torch.load(name))

        model.eval()

        Preds = torch.zeros(cfg.ts_batch_size, 1).to(cfg.device)

        for X in dataloader:

          X = X.to(cfg.device)

          with torch.no_grad():

            logProb = model(X)

          Prob = torch.exp(logProb)

          Pred = torch.argmax(Prob, dim=1)

          Preds = torch.cat((Preds, Pred.view(-1,1)), dim=0)

    Preds = Preds[cfg.ts_batch_size:].cpu().numpy()

    final_Pred = np.add(final_Pred, Preds)

    final_Pred = np.divide(final_Pred, len(names))

    return Preds.reshape(-1)
LP, Names, Network = train_fn(X, y)



LPV = []

LPT = []

for i in range(len(LP)):

  LPT.append(LP[i]["train"])

  LPV.append(LP[i]["valid"])
xAxis = [i+1 for i in range(cfg.epochs)]



plt.figure(figsize=(15, 20))

plt.subplot(4, 1, 1)

plt.plot(xAxis, LPT[:cfg.epochs], "b", label="Training Loss")

plt.plot(xAxis, LPV[:cfg.epochs], "r--", label="Validation Loss")

plt.legend()

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.subplot(4, 1, 2)

plt.plot(xAxis, LPT[1*cfg.epochs:2*cfg.epochs], "b", label="Training Loss")

plt.plot(xAxis, LPV[1*cfg.epochs:2*cfg.epochs], "r--", label="Validation Loss")

plt.legend()

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.subplot(4, 1, 3)

plt.plot(xAxis, LPT[2*cfg.epochs:3*cfg.epochs], "b", label="Training Loss")

plt.plot(xAxis, LPV[2*cfg.epochs:3*cfg.epochs], "r--", label="Validation Loss")

plt.legend()

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.subplot(4, 1, 4)

plt.plot(xAxis, LPT[3*cfg.epochs:4*cfg.epochs], "b", label="Training Loss")

plt.plot(xAxis, LPV[3*cfg.epochs:4*cfg.epochs], "r--", label="Validation Loss")

plt.legend()

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.show()
ts_data_setup = DS(X_test, None, "test")

ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)



y_pred = eval_fn(Network, Names, ts_data)

y_pred = y_pred.astype(int)



ss = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

ss["Label"] = y_pred

ss.to_csv("./submission.csv", index=False)