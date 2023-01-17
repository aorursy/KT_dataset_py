import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')
df_names = ['train_features', 'train_targets_scored', 'train_targets_nonscored', 'test_features', 'sample_submission']

df = {}
for name in df_names:
    df[name] = pd.read_csv(f"../input/lish-moa/{name}.csv", index_col=0)
    print(f"{name}: {df[name].shape}")
df_full = pd.concat([df['train_features'], df['train_targets_scored'].add_prefix('scored_'), df['train_targets_nonscored'].add_prefix('nonscored_')], axis=1)
df_full['count_scored'] = df_full.filter(regex='^scored_').sum(axis=1)
df_full['count_nonscored'] = df_full.filter(regex='^nonscored_').sum(axis=1)
df_full['count_total'] = df_full['count_scored'] + df_full['count_nonscored']

total_target_cols = list(df_full.filter(regex='^(scored|nonscored)_').columns)
print("length of total targets, scored and non scored:", len(total_target_cols))
df_target_combine = df_full[["cp_type"]+total_target_cols].groupby(total_target_cols).size().reset_index(name='Combination_Count').sort_values('Combination_Count', ascending=False).reset_index(drop=True)
df_target_combine['Combination_Label'] = df_target_combine.index
df_target_combine = df_target_combine[['Combination_Label', 'Combination_Count'] + total_target_cols]
df_target_combine.head()
transition = df_target_combine[['Combination_Label']+total_target_cols].set_index('Combination_Label')
transition.head()
df_full_combine = pd.merge(df_full, df_target_combine, on=total_target_cols, how='left')
df_full_combine.head()
df_full_trt = df_full_combine[df_full_combine.cp_type !='ctl_vehicle']
df_train = df_full_trt.drop(columns= total_target_cols+['count_scored',
       'count_nonscored', 'count_total', 
       'Combination_Count'] ).reset_index(drop = True).drop('cp_type', axis = 1)

#df_test = df['test_features'][df['test_features']['cp_type']!='ctl_vehicle'].reset_index(drop = True).drop('cp_type', axis = 1)
df_test = df['test_features'].drop('cp_type', axis = 1)
combination_label_target = df_full_trt.Combination_Label.reset_index(drop = True)
df_target_OHE = pd.get_dummies(combination_label_target, prefix='combo')
df_train.head()
df_target_OHE.head()
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle= True)
folds = df_train.copy()

for f, (train_index, val_index)  in enumerate(skf.split(X= df_train, y=combination_label_target)):
    folds.loc[val_index, 'kfold'] = int(f)
    
folds['kfold'] =folds['kfold'].astype(int)
folds.head()

class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    #the __len__ and __getitem__ are for torch.utils.data.DataLoader to load batches into neural networks.    
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
    

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    # this is to tell the model that it is in train mode, thus use batch normalization and dropout.
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    #model.eval() is to disable batch normalization and dropout.
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
        
        #this saves memory and accelerate the running time as we don't need to keep memory for the gradients.
        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds
   
    
class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2)
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
def process_data(data):
    
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
    return data
feature_cols = [c for c in process_data(folds).columns if c!='Combination_Label']
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
target_cols = [c for c in df_target_OHE.columns]
print(len(feature_cols), len(target_cols))
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=2020)
# HyperParameters

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 20
EARLY_STOP = False

num_features=len(feature_cols)
num_targets=df_target_OHE.shape[1]
hidden_size=1024
#need to modify the dataframe names
def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = process_data(folds)
    test_ = process_data(df_test)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    train_target = df_target_OHE.iloc[trn_idx,:]
    val_target = df_target_OHE.iloc[val_idx,:]
    
    #x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    #x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values

    x_train, y_train = train_df[feature_cols].values, train_target.values
    x_valid, y_valid =  valid_df[feature_cols].values,val_target.values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #optimizer = torch.optim.RMsprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
    oof = np.zeros((len(df_train), df_target_OHE.shape[1]))
    best_loss = np.inf
    
    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer,scheduler, loss_fn, trainloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"FOLD{fold}_.pth")
        
        elif(EARLY_STOP == True):
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                break
            
    """"
    as a first step we train the combination targets and
    just put the combination back using transition matrix of size (696* 608)
    
    Second step would be to use transition targets
    """
    
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    #predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = np.zeros((len(test_), df_target_OHE.shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions
   
def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(df_train), df_target_OHE.shape[1]))
    predictions = np.zeros((len(df_test), df_target_OHE.shape[1] ))
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions
SEED = [42,2020,2,27]
#SEED = [2]
oof = np.zeros((len(df_train),df_target_OHE.shape[1] ))
predictions = np.zeros((len(df_test), df_target_OHE.shape[1]))

for seed in SEED:
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)

#train[target_cols] = oof
#test[target_cols] = predictions


#normalize probabilities
for i in range(oof.shape[0]):
    oof[i,:] = oof[i,:]/np.sum(oof[i,:])
    
oof_original_target = np.matmul(oof, transition.to_numpy() )
prediction_original_target = np.matmul(predictions,transition.to_numpy())
#calculate train log loss
train_log_loss = 0
num_of_scored_targets = df['train_targets_scored'].shape[1]

trt_index = df_full_combine[df_full_combine.cp_type !='ctl_vehicle'].index
for i in range(num_of_scored_targets):
    y_true = df['train_targets_scored'].iloc[trt_index,i].values
    y_pred = oof_original_target[:,i]
    train_log_loss += log_loss(y_true, y_pred)/num_of_scored_targets
train_log_loss
prediction_original_target.shape
scored_target_cols = df['sample_submission'].columns
sub = df['sample_submission'].copy()
sub[scored_target_cols] = prediction_original_target[:,:len(scored_target_cols)]

df['test_features'].reset_index(drop = True, inplace = True)
test_ctrl_index = df['test_features'][df['test_features'].cp_type == 'ctl_vehicle'].index

sub.iloc[test_ctrl_index,:] = 0
def prob_clip(df, floor, ceiling):
    for c in range(df.shape[1]):
        df.iloc[:,c] = np.where(df.iloc[:,c]< bar, bar, df.iloc[:,c])
        df.iloc[:,c] = np.where(df.iloc[:,c] > ceiling, ceiling, df.iloc[:,c])
    return df
sub_clipped = prob_clip(sub, 0.00015, 0.995)
sub_clipped.to_csv('submission.csv',index = True)
