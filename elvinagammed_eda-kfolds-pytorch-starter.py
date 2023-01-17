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
train_targets_scored = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")
train_features = pd.read_csv("../input/lish-moa/train_features.csv")
test_features = pd.read_csv("../input/lish-moa/test_features.csv")
sample_submission = pd.read_csv("../input/lish-moa/sample_submission.csv")
train_features.shape
train_features
train_features['sig_id'].nunique()
train_features.cp_type.value_counts()
train_features.cp_time.value_counts()
train_features.cp_dose.value_counts()
train_targets_scored.head()
train_targets_scored.sum()[1:].sort_values()
train_features[:2]
gs = train_features[:1][[col for col in train_features.columns if 'g-' in col]].values.reshape(-1, 1)
import matplotlib.pyplot as plt
plt.plot(gs)
plt.plot(sorted(gs))
train_features['c-0'].plot(kind='hist')
!pip install iterative-stratification
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

df = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
df.loc[:, "kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
targets = df.drop('sig_id', axis=1).values
mskf = MultilabelStratifiedKFold(n_splits=5)
for fold_, (trn_, val_) in enumerate(mskf.split(X=df, y=targets)):
    df.loc[val_, "kfold"] = fold_

df.to_csv("train_folds.csv", index=False)
import torch
import torch.nn
class MoaDataset:
    def __init__(self, dataset, features):
        self.dataset = dataset
        self.features = features
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, item):
        return {
            "x": torch.tensor(self.dataset[item, :], dtype=torch.float),
            "y": torch.tensor(self.features[item, :], dtype=torch.float)
        }
class Engine:
#     Model, optimizer, and device are fixed, 
#     thus, they are in the init function
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)
#     data (batches) can change, thus, data, and model..., 
#     are in different functions
    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

#     validation
    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            final_loss += loss.item()
        return final_loss / len(data_loader)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    print('using device: cuda')
else:
    print('using device: cpu')
class Model(nn.Module):
    def __init__(self, num_features, num_targets):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNormalization(256),
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.BatchNormalization(256),
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.BatchNormalization(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_targets)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
DEVICE = "cuda"
EPOCHS = 100
def add_dummies(data, column):
    ohe = pd.get_dummies(data[column])
    ohe_columns = [f"{column}_{c}" for c in ohe.columns]
    ohe.columns = ohe_columns
    data = data.drop(column, axis=1)
    data = data.join(ohe)
    return data
def process_data(df):
    df = add_dummies(df, "cp_time")
    df = add_dummies(df, "cp_dose")
    df = add_dummies(df, "cp_type")
    return df
def run_training(fold):
    df = pd.read_csv("../input/lish-moa/train_features.csv")
    df = process_data(df)
    folds = pd.read_csv("../working/train_folds.csv")
    
    targets = folds.drop(["sig_id", "kfold"], axis=1).columns
    features = df.drop("sig_id", axis=1).columns
    
    df = df.merge(folds, on="sig_id", how="left")
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    
    x_train = train_df[features].to_array()
    x_valid = valid_df[features].to_array()
                     
    y_train = train_df[features].to_array()
    y_valid = valid_df[targets].to_array()
    
    train_dataset = MoaDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, num_workers=8
    )
                     
    valid_dataset = MoaDataset(x_valid, y_valid)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1024, num_workers=8
    )
    model = utils.ModelX(
        num_features = x_train.shape[1],
        num_targets = y_train.shape[1],
        num_layers = params["num_layers"],
        hidden_size = params["hidden_size"],
        dropout =  params["dropout"]
    )
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, threshold=0.00001, mode="min", verbose=True)
                     
    eng = Engine(
        model, optimizer, device=DEVICE
    )
    best_loss = np.inf
    early_stopping = 10
    early_stopping_counter = 0
    for _ in range(EPOCHS):
        train_loss = engine.train(train_loader)
        valid_loss = engine.train(valid_loader)
        scheduler.step(valid_loss)
        print(f"{fold}, {epoch}, {train_loss}, {valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), f"model{fold}.bin")
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping:
            break
    return best_loss

import optuna
class ModelX(nn.Module):
    def __init__(self, num_features, num_targets, num_layers, hidden_size, dropout):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(num_features, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                nn.ReLU()
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                nn.ReLU()
        layers.append(nn.Linear(hidden_size, num_targets))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.model(x)
        return x
