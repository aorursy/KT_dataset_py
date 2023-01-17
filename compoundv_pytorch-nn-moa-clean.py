# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# !pip install optuna --user
# import optuna
import sys
sys.path.append("../input/iterativestratification/iterative-stratification-master/")
sys.path.append("../input/my-lib-pytorch-nn/lib_kaggle/")
# Modified from
# https://www.kaggle.com/utkukubilay/notebooks

import sys

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# from lib.model import Model
# from lib.model import train_fn, valid_fn, inference_fn
from lib.data import *
from lib.data_processing import *

import numpy as np
import random
import pandas as pd
import os
import json


from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
# model definition 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    count = 0
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        # print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        # change the lr if use OneCycle schedule
        # scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)
    scheduler.step()

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds
data_loc = "../input/lish-moa/"
# load data
train_features = pd.read_csv(data_loc + 'train_features.csv')
train_targets_scored = pd.read_csv(data_loc + 'train_targets_scored.csv')
# train_targets_nonscored = pd.read_csv(data_loc +  'train_targets_nonscored.csv')

test_features = pd.read_csv(data_loc + 'test_features.csv')
# sample_submission = pd.read_csv(data_loc + 'sample_submission.csv')

target_cols = [x for x in train_targets_scored.columns.to_list() if x != 'sig_id']
# save the target_cols into json file

print(train_features.shape, test_features.shape)
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
# ohe

data_entire = pd.concat([train_features, test_features], axis=0)

# name_dict = {}
# for i, col in enumerate(train_features.columns.to_list()):
#     name_dict[i] = col
    
# # data_train = data_final.iloc[0:23814, :]
data_entire.iloc[23812:23816, :]
feature_to_scale = ["cp_time"] + GENES + CELLS
print(len(feature_to_scale))

data_scaler = MinMaxScaler()
data_entire[feature_to_scale ] = data_scaler.fit_transform(data_entire[feature_to_scale ])
# data_entire[feature_to_scale].describe()
print(data_entire.shape)
# ohe
cols_ohe = ['cp_type', 'cp_dose']
data_entire = ohe(data_entire, cols_ohe)

data_entire.head(2)
data_train = data_entire.iloc[0:23814, :].copy(deep=True)
data_test = data_entire.iloc[23814:, :].copy(deep=True)
tmp_data_dir = "/kaggle/working/tmp_data/"
try:
    os.makedirs(tmp_data_dir)
except FileExistsError:
    pass

data_train.to_pickle(tmp_data_dir + "data_train.pkl")
data_test.to_pickle(tmp_data_dir + "data_test.pkl")
seed_everything(seed=1903)

tmp_data_dir = "/kaggle/working/tmp_data/"

data_train_x = pd.read_pickle(tmp_data_dir + "data_train.pkl")
data_test_x = pd.read_pickle(tmp_data_dir + "data_test.pkl")

target_cols = [x for x in train_targets_scored.columns.to_list() if x != 'sig_id']
feature_cols = [c for c in data_train_x.columns.to_list() if c != 'sig_id']
print(len(feature_cols)); print(len(target_cols))
train_entire = data_train_x.merge(train_targets_scored, on="sig_id")
# ==========================
# # k-fold split
#
NFOLDS = 5
seed = 1903
mskf = MultilabelStratifiedKFold(n_splits=NFOLDS, random_state=seed)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=data_train_x, y=train_targets_scored[target_cols])):
    train_entire.loc[v_idx, 'kfold'] = int(f)

train_entire['kfold'] = train_entire['kfold'].astype(int)
train_entire.to_pickle(tmp_data_dir + "data_train_process2_5fold_1903.pkl")
# ============================
def run_training(train, valid, feature_cols, target_cols, model_path, seed, param_provided=None):
    seed_everything(seed)

    x_train, y_train = train[feature_cols].values, train[target_cols].values
    
    # create the dataset loader
    train_dataset = MoADataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    if valid is not None:
        x_valid, y_valid = valid[feature_cols].values, valid[target_cols].values
        valid_dataset = MoADataset(x_valid, y_valid)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)



    # create an model instance
    if param_provided is not None:
        EPOCHS = param_provided['epoch']
        hidden_size = param_provided['hidden_size']
        LEARNING_RATE = param_provided['lr']
    
    print("hidden_size: ", hidden_size, ", learning_rate: ", LEARNING_RATE)
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.05, div_factor=1.5e3,
    #                                           max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    
    loss_fn = nn.BCEWithLogitsLoss()

    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    best_loss = np.inf

    #
    train_losses = []; valid_losses = []

    for epoch in range(EPOCHS):

        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
        train_losses.append(train_loss)
        # print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        
        if valid is not None: # only run the valid if valid set is provided
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
            valid_losses.append(valid_loss)

            if epoch % 5 == 0:
                print(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")

            if valid_loss < best_loss:

                best_loss = valid_loss
                # oof[val_idx] = valid_preds
                torch.save(model.state_dict(), model_path)

            elif EARLY_STOP == True:
                early_step += 1
                if early_step >= early_stopping_steps:
                    break
        else:
            if epoch % 10 == 0:
                print(f"EPOCH: {epoch}, train_loss: {train_loss}")
    
    print("early stop with epoch: ", epoch)
    print(f"LAST EPOCH: {epoch}, train_loss: {train_loss}")
    
    if valid is None: # when there is not valid set, save the model
        torch.save(model.state_dict(), model_path)

    return {"train_losses": train_losses, "valid_losses": valid_losses}
def run_training_tune(trial, train, valid, feature_cols, target_cols, model_path, seed):
    seed_everything(seed)

    x_train, y_train = train[feature_cols].values, train[target_cols].values
    x_valid, y_valid = valid[feature_cols].values, valid[target_cols].values

    # create the dataset loader
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    hidden_size = trial.suggest_int("hidden_size", 200, 500, step=50)
    # create an model instance
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )

    model.to(DEVICE)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # lr = 0.0084

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    # this is used to change the learning rate when training model
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.05,
    #                                          max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    loss_fn = nn.BCEWithLogitsLoss()

    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    best_loss = np.inf

    #
    train_losses = []; valid_losses = []

    for epoch in range(EPOCHS):

        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
        train_losses.append(train_loss)
        # print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")

        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        valid_losses.append(valid_loss)

        if epoch % 5 == 0:
            print(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")

        if valid_loss < best_loss:

            best_loss = valid_loss
            # now don't save the model to adapt to the use of optuna
            # torch.save(model.state_dict(), model_path)

        elif EARLY_STOP == True:
            early_step += 1
            if early_step >= early_stopping_steps:
                break
                
#     don't need this at the moment
#     if trial.should_prune():
#         raise optuna.exceptions.TrialPruned()

    return valid_losses[-1]

def print_model_results(train_loss, valid_loss, train_len, valid_len):
    total_num_inst = train_len + valid_len
    train_loss_final = train_loss
    valid_loss_final = valid_loss
    entire_train_loss = train_len / total_num_inst * train_loss_final \
                        + valid_len / total_num_inst * valid_loss_final
    print("train loss: ", train_loss_final, " valid loss: ", valid_loss_final,
          "entire loss: ", entire_train_loss)
import datetime
import time

start_time = datetime.datetime.now()
time.sleep(5)
end_time = datetime.datetime.now()
print("time elapsed: ", end_time -  start_time)
# # run the main function

# seed_everything(seed=1903)

# # data_loc = "../input/lish-moa/"
# # # load data
# # train_features = pd.read_csv(data_loc + 'train_features.csv')
# # train_targets_scored = pd.read_csv(data_loc + 'train_targets_scored.csv')
# # # train_targets_nonscored = pd.read_csv(data_loc +  'train_targets_nonscored.csv')
# #
# # test_features = pd.read_csv(data_loc + 'test_features.csv')
# # # sample_submission = pd.read_csv(data_loc + 'sample_submission.csv')
# #
# # target_cols = [x for x in train_targets_scored.columns.to_list() if x != 'sig_id']
# # # save the target_cols into json file
# #
# #
# # # prepare training/test data
# # data_train_x, data_test_x = one_step_processing(train_features, test_features)
# #
# # feature_cols = [c for c in data_train_x.columns.to_list() if c != 'sig_id']

# # ====
# # with open("../input/process_1/target_cols.json", "w", encoding="utf-8") as f:
# #     json.dump(target_cols, f)
# # with open("../input/process_1/feature_cols.json", "w", encoding="utf-8") as f:
# #     json.dump(feature_cols, f)

# # data_processed_input = "../input/process-1/process_1/"
# # with open(data_processed_input + "target_cols.json", "r", encoding="utf-8") as f:
# #     target_cols = json.load(f)
# # with open(data_processed_input + "feature_cols.json", "r", encoding="utf-8") as f:
# #     feature_cols = json.load(f)


# # HyperParameters
# # ==========================
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(DEVICE)
# EPOCHS = 100 # 100
# # EPOCHS = 2
# BATCH_SIZE = 64
# # LEARNING_RATE = 1e-3
# LEARNING_RATE = 1e-2
# WEIGHT_DECAY = 1e-5

# EARLY_STOPPING_STEPS = 11
# EARLY_STOP = True

# num_features = len(feature_cols)
# num_targets = len(target_cols)
# # hidden_size = 1024
# hidden_size = 50

# if_tune = True
# # =====================================================
# # run kFold (# 3) twice
# # SEED = [1903, 1881]
# SEED = [1903]


# prefix = "process_1"

# models_dir = "/kaggle/working/trained_models/"
# try:
#     os.makedirs(models_dir)
# except FileExistsError:
#     pass


# for seed in SEED:
#     NFOLDS = 5
#     # step 1: do the splitting on train_entire here to generate train, valid

#     # train_entire = data_train_x.merge(train_targets_scored, on="sig_id")
#     # # ==========================
#     # # # k-fold split
#     # #
#     # NFOLDS = 5
#     # mskf = MultilabelStratifiedKFold(n_splits=NFOLDS, random_state=seed)
#     #
#     # for f, (t_idx, v_idx) in enumerate(mskf.split(X=data_train_x, y=train_targets_scored[target_cols])):
#     #     train_entire.loc[v_idx, 'kfold'] = int(f)
#     #
#     # train_entire['kfold'] = train_entire['kfold'].astype(int)
#     # # we can create 5 train-valid pairs
#     # train = train_entire[train_entire["kfold"] != 4]
#     # valid = train_entire[train_entire["kfold"] == 4]
#     # ==========================
#     # train-valid split
#     # msss = MultilabelStratifiedShuffleSplit(nsplit=2, test_size=0.2,  random_state=seed)
#     # for train_index, test_index in msss.split(X=data_train_x, y=train_targets_scored[target_cols]):
#     #     train, valid = train_entire[train_index], train_entire[test_index]

#     # ==========================

#     # one can just load the post-process data here
#     # train_entire = pd.read_pickle(data_processed_input + "data_train_5fold_1903.pkl")
#     train_entire = pd.read_pickle(tmp_data_dir + "data_train_process2_5fold_1903.pkl")
#     best_param = {}
    
#     for k_fold in np.arange(NFOLDS):
#     # for k_fold in [4]:
#         print("runing fold ", k_fold)
#         # k_fold = 4
#         train = train_entire[train_entire["kfold"] != k_fold]
#         valid = train_entire[train_entire["kfold"] == k_fold]
#         # data_test_x = pd.read_pickle(data_processed_input + "data_test_x.pkl")
#         data_test_x = pd.read_pickle(tmp_data_dir + "data_test.pkl")
        
#         model_path = models_dir + prefix + "_model_fold_{0:d}.pth".format(k_fold)

#         model_result = None
        
#         if if_tune:
#             # optuna for hyperparameter tuning
#             # ==========================
#             start_time = datetime.datetime.now()

#             study = optuna.create_study(direction="minimize")
#             study.optimize(lambda trial: run_training_tune(trial, train, valid, feature_cols, target_cols, model_path,
#                                                        seed), n_trials=20) # timeout=600
            
#             # to check the optuna optimization
#             pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
#             complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
#             print("Study statistics: ")
#             print("  Number of finished trials: ", len(study.trials))
#             print("  Number of pruned trials: ", len(pruned_trials))
#             print("  Number of complete trials: ", len(complete_trials))

#             print("Best trial:")
#             trial = study.best_trial

#             print("  Value: ", trial.value)

#             print("  Params: ")
#             for key, value in trial.params.items():
#                 print("    {}: {}".format(key, value))
                
#             best_param["kfold_" + str(k_fold)] = trial.params
            
#             end_time = datetime.datetime.now()
#             print("time elapsed: ", end_time -  start_time)
            
#         else:
#             model_result = run_training(train, valid, feature_cols, target_cols, model_path, seed)
            
#             print_model_results(model_result["train_losses"][-1], model_result["valid_losses"][-1],
#                     train.shape[0], valid.shape[0])
best_param_with_epoch = {'kfold_0': {'hidden_size': 450, 'lr': 0.0008925929114331024, 'epoch': 30},
 'kfold_1': {'hidden_size': 400, 'lr': 0.0051852480304641155, 'epoch': 17},
 'kfold_2': {'hidden_size': 200, 'lr': 0.0020302278497303823, 'epoch': 30},
 'kfold_3': {'hidden_size': 300, 'lr': 0.0029643441803447576, 'epoch': 20},
 'kfold_4': {'hidden_size': 250, 'lr': 0.006931570763035235, 'epoch': 16}}

# run the main function

seed_everything(seed=1903)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
EPOCHS = 100 # 100
# EPOCHS = 2
BATCH_SIZE = 64
# LEARNING_RATE = 1e-3
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5

EARLY_STOPPING_STEPS = 11
EARLY_STOP = True

num_features = len(feature_cols)
num_targets = len(target_cols)
# hidden_size = 1024
hidden_size = 50

if_tune = False
# =====================================================
# run kFold (# 3) twice
# SEED = [1903, 1881]
SEED = [1903]


prefix = "process_2"

models_dir = "/kaggle/working/trained_models/"
try:
    os.makedirs(models_dir)
except FileExistsError:
    pass

# let's find the best epoch first
for seed in SEED:
    NFOLDS = 5
    train_entire = pd.read_pickle(tmp_data_dir + "data_train_process2_5fold_1903.pkl")
    best_param = {}
    
    for k_fold in np.arange(NFOLDS):
    # for k_fold in [0]:
        print("runing fold ", k_fold)
        
#         print("hyperparameters: ", best_param_with_epoch["kfold_" + str(k_fold)])
#         EPOCHS = best_param_with_epoch[key]['epoch']
#         EPOCHS = 100
#         hidden_size = best_param_with_epoch[key]['hidden_size']
#         LEARNING_RATE = best_param_with_epoch[key]['lr']
        
        # k_fold = 4
        # train = train_entire[train_entire["kfold"] != k_fold]
        # valid = train_entire[train_entire["kfold"] == k_fold]
        train = train_entire
        valid = None
        
        # data_test_x = pd.read_pickle(data_processed_input + "data_test_x.pkl")
        data_test_x = pd.read_pickle(tmp_data_dir + "data_test.pkl")
        
        model_path = models_dir + prefix + "_model_fold_{0:d}.pth".format(k_fold)

        model_result = None
        
        if if_tune:
            # optuna for hyperparameter tuning
            # ==========================
            start_time = datetime.datetime.now()

            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: run_training_tune(trial, train, valid, feature_cols, target_cols, model_path,
                                                       seed), n_trials=20) # timeout=600
            
            # to check the optuna optimization
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                
            best_param["kfold_" + str(k_fold)] = trial.params
            
            end_time = datetime.datetime.now()
            print("time elapsed: ", end_time -  start_time)
            
        else:
            start_time = datetime.datetime.now()
            model_result = run_training(train, valid, feature_cols, target_cols, 
                                        model_path, seed, 
                                        param_provided = best_param_with_epoch["kfold_" + str(k_fold)])
            end_time = datetime.datetime.now()
            print("time elapsed: ", end_time -  start_time)
            
            if valid is None:
                print("loss on the entire training data", model_result["train_losses"][-1])
            else:
                print_model_results(model_result["train_losses"][-1], model_result["valid_losses"][-1],
                        train.shape[0], valid.shape[0])
# --------------------- PREDICTION on test dataset---------------------
x_test = data_test_x[feature_cols].values
testdataset = TestDataset(x_test)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

for k_fold in np.arange(NFOLDS):
# for k_fold in [0]:
    
    if best_param_with_epoch is not None:
        hidden_size = best_param_with_epoch["kfold_" + str(k_fold)]['hidden_size']
        
    model_path = models_dir + prefix + "_model_fold_{0:d}.pth".format(k_fold)
        
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    
    predictions = inference_fn(model, testloader, DEVICE)
    # save the prediction on test set
    pred_test = pd.DataFrame(data=predictions, columns=target_cols).fillna(0)
    pred_test = pd.concat([data_test_x[["sig_id"]], pred_test], axis=1)
    pred_result_f = models_dir + prefix + "_fold_{0:d}_prediction.csv".format(k_fold)
    pred_test.to_csv(pred_result_f, index=False)
# for k_fold in np.arange(NFOLDS):
folds_to_use = np.arange(NFOLDS)
for i_count, k_fold in enumerate(folds_to_use):
    pred_result_f = models_dir + prefix + "_fold_{0:d}_prediction.csv".format(k_fold)

    pred_test = pd.read_csv(pred_result_f, header=0)
    pred_test[target_cols] = pred_test[target_cols]/len(folds_to_use)
    if i_count == 0:
        pred_test_average = pred_test.copy(deep=True)
    else:
        pred_test_average[target_cols] = pred_test_average[target_cols] + pred_test[target_cols]
sample_submission = pd.read_csv("../input/lish-moa/" + 'sample_submission.csv')

sub = sample_submission.drop(columns=target_cols).merge(pred_test_average, on='sig_id', how='left').fillna(0)

sub.to_csv('submission.csv', index=False)
