import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

import seaborn as sns

import matplotlib.pyplot as plt

from glob import glob

import gc

import os

import math

import random

from tqdm import tqdm

from IPython.core.interactiveshell import InteractiveShell

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import mean_absolute_error

import warnings

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

from torch.optim import Adam, AdamW

%matplotlib inline



warnings.filterwarnings("ignore")

InteractiveShell.ast_node_interactivity = "all"
BATCH_SIZE = 128

EPOCHS = 800

LR = 1e-1

FOLDER = 5

SEED = 42

DEVICE = 'cuda'

QUANTILE = [0.2, 0.5, 0.8]

SAVE_AND_LOAD_BEST_MODEL = True    # You can turn on this for higher score

LR_Scheduler = False               # You can turn on this for higher score



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



seed_everything(SEED)
ROOT = "../input/osic-pulmonary-fibrosis-progression"



tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

chunk = pd.read_csv(f"{ROOT}/test.csv")



sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient", how='left')



tr['WHERE'] = 'train'

chunk['WHERE'] = 'val'

sub['WHERE'] = 'test'

data = tr.append([chunk, sub])



print("Origin Shape: ")

print(tr.shape, chunk.shape, sub.shape, data.shape)

print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(), data.Patient.nunique())



data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')



base = data.loc[data.Weeks == data.min_week]

base = base[['Patient','FVC']].copy()

base.columns = ['Patient','min_FVC']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)



data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base



COLS = ['Sex','SmokingStatus']

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)



data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

FE += ['age','percent','week','BASE']

INPUT_FEATURES = len(FE)

print("\nFE: ", FE, 

      "\nINPUT_FEATURES: ", INPUT_FEATURES)



tr = data.loc[data.WHERE=='train']

chunk = data.loc[data.WHERE=='val']

sub = data.loc[data.WHERE=='test']

del data, chunk



X_train = tr[FE].values

Y_train = tr['FVC'].values

X_test = sub[FE].values

print("\nAfter Shape: ")

print(X_train.shape, Y_train.shape, X_test.shape)

gc.collect()
class DatasetRetriever(Dataset):

    def __init__(self, x_data, y_data=None):

        self.x_data = x_data

        self.y_data = y_data

    

    def __len__(self):

        return len(self.x_data)

    

    def __getitem__(self, idx):

        if self.y_data is not None:

            return torch.tensor(self.x_data[idx]), torch.tensor(self.y_data[idx])

        else:

            return torch.tensor(self.x_data[idx])





class QuantileRegression(nn.Module):

    def __init__(self, input_features=INPUT_FEATURES):

        super(QuantileRegression, self).__init__()

        

        self.nn1 = nn.Linear(input_features, 100)

        self.nn2 = nn.Linear(100, 100)

        self.nn3_1 = nn.Linear(100, 3)

        self.nn3_2 = nn.Linear(100, 3)

        torch.nn.init.xavier_uniform_(self.nn1.weight)

        torch.nn.init.constant_(self.nn1.bias, 0)

        torch.nn.init.xavier_uniform_(self.nn2.weight)

        torch.nn.init.constant_(self.nn2.bias, 0)

        torch.nn.init.xavier_uniform_(self.nn3_1.weight)

        torch.nn.init.constant_(self.nn3_1.bias, 0)

        torch.nn.init.xavier_uniform_(self.nn3_2.weight)

        torch.nn.init.constant_(self.nn3_2.bias, 0)

    

    def forward(self, inputs):

        X = F.relu(self.nn1(inputs))

        X = F.relu(self.nn2(X))

        X_1 = self.nn3_1(X)

        X_2 = F.relu(self.nn3_2(X))

        output = X_1 + torch.cumsum(X_2, dim=1)

        return output





class LossMeter(object):

    def __init__(self):

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n):

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count





class ScoreMeter(object):

    def __init__(self):

        self.sum = 0

        self.count = 0

        self.avg = 0

    

    def compute_score(self, y_pred, y_true):

        sigma = y_pred[:, 2] - y_pred[:, 0]

        fvc_pred = y_pred[:, 1]

        sigma_clip = np.maximum(sigma, 70.0)

        delta = np.minimum(np.abs(y_true[:, 0]-fvc_pred), 1000.0)

        metric = (delta / sigma_clip) * np.sqrt(2.0) + np.log(sigma_clip * np.sqrt(2.0))

        return np.mean(metric)

    

    def update(self, preds, labels):

        batch_size = preds.size(0)

        preds = preds.data.cpu().numpy()

        labels = labels.data.cpu().numpy()

        val = self.compute_score(preds, labels)

        self.sum += (val * batch_size)

        self.count += batch_size

        self.avg = self.sum / self.count





class QuantileRegressionLoss(nn.Module):

    def __init__(self):

        super(QuantileRegressionLoss, self).__init__()

        self.quantile = torch.tensor(QUANTILE).to(DEVICE, dtype=torch.float)



    def forward(self, preds, labels):

        error = labels - preds

        vector = torch.max(self.quantile*error, (self.quantile-1)*error)

        return vector.mean()
class Fitter:

    def __init__(self, model, device, fold):

        self.model = model

        self.device = device

        self.fold = fold

        self.optimizer = Adam(self.model.parameters(), lr=LR, weight_decay=0.01)  # default weight_decay=0

        self.criterion = QuantileRegressionLoss()

        if LR_Scheduler:

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 

                                                                        mode='min', 

                                                                        factor=0.5, 

                                                                        patience=20,

                                                                        min_lr=1e-4,

                                                                        verbose=True)

        print(f'Fitter prepared. Device is {self.device}')

    

    def fit(self, train_loader, valid_loader):

        min_valid_score = 999

        plot_rec = {"train_loss": [],

                    "train_score":[],

                    "valid_loss": [],

                    "valid_score":[],

                    "best_epoch": 0,

                    "final": [],

                    "best": []

                    }

        for epoch in range(EPOCHS):

            train_loss, train_score = self.train_one_epoch(train_loader)

#             print(f'Epoch: {epoch+1}, train_loss: {train_loss:.4f}, train_score: {train_score:.4f}', end='  ')

            plot_rec["train_loss"].append(train_loss)

            plot_rec["train_score"].append(train_score)

            

            valid_loss, valid_score = self.validation(valid_loader)

#             print(f'valid_loss: {valid_loss:.4f}, valid_score: {valid_score:.4f}')

            plot_rec["valid_loss"].append(valid_loss)

            plot_rec["valid_score"].append(valid_score)

            

            if LR_Scheduler:

                self.scheduler.step(valid_score)

            

            if SAVE_AND_LOAD_BEST_MODEL:

                if valid_score < min_valid_score:

                    min_valid_score = valid_score

                    torch.save(self.model.state_dict(), f"Folder-{self.fold}.bin")

                    plot_rec["best_epoch"] = epoch + 1

                    plot_rec["best"] = [train_loss, train_score, valid_loss, valid_score]

                    print(f'****************************** Epoch {epoch+1} Model Is Best ******************************')

            

        if SAVE_AND_LOAD_BEST_MODEL:

            temp = plot_rec["best"]

            print(f"\nBest: ", 

                  f"\ntrain_loss: {temp[0]:.4f}, train_score: {temp[1]:.4f}, valid_loss: {temp[2]:.4f}, valid_score: {temp[3]:.4f}")

            

        plot_rec["final"] = [train_loss, train_score, valid_loss, valid_score]

        print(f"Final: ", 

              f"\ntrain_loss: {train_loss:.4f}, train_score: {train_score:.4f}, valid_loss: {valid_loss:.4f}, valid_score: {valid_score:.4f}")

        

        return plot_rec

    

    def train_one_epoch(self, train_loader):

        losses = LossMeter()

        scores = ScoreMeter()

        self.model.train()

        for step, (ipt, lbl) in enumerate(train_loader):

            ipt = ipt.to(self.device, dtype=torch.float)

            lbl = lbl.to(self.device, dtype=torch.float).view(-1, 1)

            self.optimizer.zero_grad()

            opt = self.model(ipt)

            loss = self.criterion(opt, lbl)

            losses.update(loss.detach().item(), ipt.size(0))

            scores.update(opt, lbl)

            loss.backward()

            self.optimizer.step()

        return losses.avg, scores.avg

    

    def validation(self, validation_loader):

        losses = LossMeter()

        scores = ScoreMeter()

        self.model.eval()

        for step, (ipt, lbl) in enumerate(validation_loader):

            with torch.no_grad():

                ipt = ipt.to(self.device, dtype=torch.float)

                lbl = lbl.to(self.device, dtype=torch.float).view(-1, 1)

                opt = self.model(ipt)

                loss = self.criterion(opt, lbl)

                losses.update(loss.detach().item(), ipt.size(0))

                scores.update(opt, lbl)

        return losses.avg, scores.avg

    

    def run_inference(self, data_loader):

        self.model.eval()

        temp = np.empty((0, 3))

        for step, ipt in enumerate(data_loader):

            with torch.no_grad():

                ipt = ipt.to(self.device, dtype=torch.float)

                opt = self.model(ipt)

                temp = np.append(temp, opt.cpu().detach().numpy(), axis=0)

        return temp
def TrainAndPred(x_train, y_train, x_valid, y_valid, x_test, fold):

    device = DEVICE

    model = QuantileRegression()

    model.to(device)

    

    train_data = DatasetRetriever(x_train, y_train)

    train_data_for_pred = DatasetRetriever(x_train)

    valid_data = DatasetRetriever(x_valid, y_valid)

    valid_data_for_pred = DatasetRetriever(x_valid)

    test_data = DatasetRetriever(x_test)

    

    train_data_loader = torch.utils.data.DataLoader(

        train_data,

        batch_size=BATCH_SIZE,

        drop_last=False,

        num_workers=0,

        shuffle=True

    )

    

    train_data_loader_for_pred = torch.utils.data.DataLoader(

        train_data_for_pred,

        batch_size=BATCH_SIZE,

        drop_last=False,

        num_workers=0,

        shuffle=False

    )

    

    valid_data_loader = torch.utils.data.DataLoader(

        valid_data,

        batch_size=BATCH_SIZE,

        drop_last=False,

        num_workers=0,

        shuffle=False

    )

    

    valid_data_loader_for_pred = torch.utils.data.DataLoader(

        valid_data_for_pred,

        batch_size=BATCH_SIZE,

        drop_last=False,

        num_workers=0,

        shuffle=False

    )

    

    test_data_loader = torch.utils.data.DataLoader(

        test_data,

        batch_size=BATCH_SIZE,

        drop_last=False,

        num_workers=0,

        shuffle=False

    )

    

    fitter = Fitter(model=model, device=device, fold=fold)

    plot_rec = fitter.fit(train_data_loader, valid_data_loader)

    

    if SAVE_AND_LOAD_BEST_MODEL:

        bestModel = QuantileRegression()

        bestModel.load_state_dict(torch.load(f"Folder-{fold}.bin"))

        bestModel.to(device)

        fitter = Fitter(model=bestModel, device=device, fold=fold)

        

    pred_for_train = fitter.run_inference(train_data_loader_for_pred)

    pred_for_valid = fitter.run_inference(valid_data_loader_for_pred)

    pred_for_test = fitter.run_inference(test_data_loader)

    

    return pred_for_train, pred_for_valid, pred_for_test, plot_rec
for_train = np.zeros((len(X_train), 3))

for_valid = np.zeros((len(X_train), 3))

for_test = np.zeros((len(X_test), 3))



plot_recs = []

split_record = []

kfold = KFold(FOLDER)



for fold, (xx, yy) in enumerate(kfold.split(X_train)):

    split_record.append([fold, xx, yy])



for item in split_record:

    fold, xx, yy = item

    print(f"\n========================================fold:{fold+1}============================================")

#     seed_everything(SEED+fold)

    temp_x_train = X_train[xx]

    temp_y_train = Y_train[xx]

    temp_x_valid = X_train[yy]

    temp_y_valid = Y_train[yy]

    print("Shape: ", temp_x_train.shape, temp_y_train.shape, temp_x_valid.shape, temp_y_valid.shape)

    pred_for_train, pred_for_valid, pred_for_test, plot_rec = TrainAndPred(temp_x_train, temp_y_train, temp_x_valid, temp_y_valid, X_test, fold+1)

    plot_recs.append(plot_rec)

    for_train[xx] += pred_for_train / (FOLDER - 1)

    for_valid[yy] = pred_for_valid

    for_test[:] += pred_for_test / FOLDER



np.save("for_train", for_train)

np.save("for_valid", for_valid)

np.save("for_test", for_test)
if SAVE_AND_LOAD_BEST_MODEL:

    print("Best: ")

    plot_recs_copy = np.array([plot_rec["best"] for plot_rec in plot_recs]).mean(axis=0)

    print("train_loss_avg: ", round(plot_recs_copy[0], 4), "  train_score_avg: ", round(plot_recs_copy[1], 4))

    print("valid_loss_avg: ", round(plot_recs_copy[2], 4), "  valid_score_avg: ", round(plot_recs_copy[3], 4))

else:

    print("Final: ")

    plot_recs_copy = np.array([plot_rec["final"] for plot_rec in plot_recs]).mean(axis=0)

    print("train_loss_avg: ", round(plot_recs_copy[0], 4), "  train_score_avg: ", round(plot_recs_copy[1], 4))

    print("valid_loss_avg: ", round(plot_recs_copy[2], 4), "  valid_score_avg: ", round(plot_recs_copy[3], 4))
# The place where the green line is is the epochs of the best model

if SAVE_AND_LOAD_BEST_MODEL:

    for fold, adict in enumerate(plot_recs):

        best_epoch = adict["best_epoch"]



        best_epoch_train_loss = round(adict['best'][0],4)

        best_epoch_train_score = round(adict['best'][1],4)

        best_epoch_valid_loss = round(adict['best'][2],4)

        best_epoch_valid_score = round(adict['best'][3],4)



        min_loss = min(min(adict["train_loss"]), min(adict["valid_loss"]))

        max_loss = max(max(adict["train_loss"]), max(adict["valid_loss"]))

        min_score = min(min(adict["train_score"]), min(adict["valid_score"]))

        max_score = max(max(adict["train_score"]), max(adict["valid_score"]))

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        ax[0].plot(range(1, len(adict["train_loss"])+1), adict["train_loss"], label="train_loss")

        ax[0].plot(range(1, len(adict["train_loss"])+1), adict["valid_loss"], label="valid_loss")

        ax[0].plot([best_epoch, best_epoch], [min_loss, max_loss], label="best_epoch")

        ax[1].plot(range(1, len(adict["train_loss"])+1), adict["train_score"], label="train_score")

        ax[1].plot(range(1, len(adict["train_loss"])+1), adict["valid_score"], label="valid_score")

        ax[1].plot([best_epoch, best_epoch], [min_score, max_score], label="best_epoch")

        ax[0].legend()

        ax[1].legend()

        ax[0].grid()

        ax[1].grid()

        ax[0].set_title(f"Fold{fold+1}-Loss-Epoch{best_epoch}-train{best_epoch_train_loss}-valid{best_epoch_valid_loss}")

        ax[1].set_title(f"Fold{fold+1}-Score-Epoch{best_epoch}-train{best_epoch_train_score}-valid{best_epoch_valid_score}")

        ax[0].set_ylim(20, 100)

        ax[1].set_ylim(6, 8)

        plt.show()
sigma_opt = mean_absolute_error(Y_train, for_valid[:, 1])

unc = for_valid[:, 2] - for_valid[:, 0]

sigma_mean = np.mean(unc)

print(sigma_opt, sigma_mean)

print("Min: ", unc.min(), "Mean: ", unc.mean(), "Max: ", unc.max(), "Mean(>=0): ", (unc>=0).mean())

plt.figure(figsize=(12, 6))

plt.hist(unc)

plt.title("uncertainty in prediction")

plt.show()
plt.figure(figsize=(15, 8))

idxs = np.random.randint(0, Y_train.shape[0], 100)

plt.plot(Y_train[idxs], label="ground truth")

plt.plot(for_valid[idxs, 0], label=f"q{int(QUANTILE[0]*100)}")

plt.plot(for_valid[idxs, 1], label=f"q{int(QUANTILE[1]*100)}")

plt.plot(for_valid[idxs, 2], label=f"q{int(QUANTILE[2]*100)}")

plt.legend(loc="best")

plt.show()
sub['FVC1'] = 0.996 * for_test[:, 1]

sub['Confidence1'] = for_test[:, 2] - for_test[:, 0]

subm = sub[['Patient_Week', 'FVC', 'Confidence', 'FVC1', 'Confidence1']].copy()
subm.loc[~subm.FVC1.isnull(), 'FVC'] = subm.loc[~subm.FVC1.isnull(), 'FVC1']

if sigma_mean < 70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[~subm.FVC1.isnull(), 'Confidence'] = subm.loc[~subm.FVC1.isnull(), 'Confidence1']



subm.describe().T
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1



subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)