import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as F
import random
import tensorflow as tf

import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

kaeru_seed = 1337
seed_everything(seed=kaeru_seed)

batch_size = 128
train_epochs = 100
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
target = train["Survived"]

train = pd.concat([train, test], sort=True)

print(train.shape)
train.head()
def get_text_features(train):
    train['Length_Name'] = train['Name'].astype(str).map(len)
    return train

train = get_text_features(train)

train.head()
cat_cols = [
     'Cabin','Embarked','Name','Sex','Ticket',
]

num_cols = list(set(train.columns) - set(cat_cols) - set(["Survived"]))
num_cols
def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]


for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(train.loc[:, cat].astype(str).unique())}
    train[cat] = train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) for encoder in encoders]
train.head()
from sklearn.preprocessing import StandardScaler
 
train[num_cols] = train[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])

train.head()
train.corr(method='pearson')
class CustomLinear(nn.Module):
    def __init__(self, in_features,
                 out_features,
                 bias=True, p=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features,out_features,bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop(x)
        return x
class TitanicModel(nn.Module):
    def __init__(self, n_input, n_hidden_neurons):
        super(TitanicModel, self).__init__()
#         self.fc1 = torch.nn.Linear(n_input, n_hidden_neurons)
#         self.activ1 = torch.nn.Sigmoid()
#         self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
#         self.activ2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_input, 1)
        self.activ3 = torch.nn.Sigmoid()
        
    def forward(self, x):
#         x = self.fc1(x)
#         x = self.activ1(x)
#         x = self.fc2(x)
#         x = self.activ2(x)
        x = self.fc3(x)
        x = self.activ3(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x
#net = nn.Sequential(CustomLinear(12, 32), nn.Linear(32, 1))
net = TitanicModel(12, 64)
X_train = train.loc[np.isfinite(train.Survived), :]
X_train = X_train.drop(["Survived"], axis=1).values
y_train = target.values

X_test = train.loc[~np.isfinite(train.Survived), :]

X_test = X_test.drop(["Survived"], axis=1).values
np.corrcoef(X_train, X_train)
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=kaeru_seed).split(X_train, y_train))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
train_preds = np.zeros((len(X_train)))
test_preds = np.zeros((len(X_test)))

seed_everything(kaeru_seed)

x_test_cuda = torch.tensor(X_test, dtype=torch.float32).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
for i, (train_idx, valid_idx) in enumerate(splits):
    x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.float32).cuda()
    y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
    x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.float32).cuda()
    y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()
    
    model = net
    model.cuda()
    
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    
    print(f'Fold {i + 1}')
    
    for epoch in range(train_epochs):
        start_time = time.time()
        
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        
        model.eval()
        valid_preds_fold = np.zeros((x_val_fold.size(0)))
        test_preds_fold = np.zeros(len(X_test))
        avg_val_loss = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
        
        elapsed_time = time.time() - start_time 
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))
        
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold / len(splits)
from sklearn.metrics import accuracy_score

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = accuracy_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'accuracy_score': best_score}
    return search_result
search_result = threshold_search(y_train, train_preds)
search_result
sub = pd.read_csv('../input/gender_submission.csv')
sub.Survived = (test_preds > search_result['threshold']).astype(np.int8)
sub.to_csv('simple_nn_submission.csv', index=False)
sub.head()