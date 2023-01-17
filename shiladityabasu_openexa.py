import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

koi_cumm_path = os.path.join('../input', 'koi-cummulative/koi_cummulative.csv')
dfc = pd.read_csv(koi_cumm_path)
dfc.shape
dfc = pd.read_csv(koi_cumm_path)
dfc['koi_disposition'].unique()
dfc['koi_pdisposition'].unique()
# 2418 candidates

(dfc['koi_disposition'] == "CANDIDATE").value_counts()
dfc.head(10)    # first 20 samples
# the columns

dfc.columns
dfc.info()
# all the non-numeric columns

df_numeric = dfc.copy()

koi_disposition_labels = {
    "koi_disposition": {
        "CONFIRMED": 1,
        "FALSE POSITIVE": 0,
        "CANDIDATE": 2,
        "NOT DISPOSITIONED": 3
    },
    "koi_pdisposition": {
        "CONFIRMED": 1,
        "FALSE POSITIVE": 0,
        "CANDIDATE": 2,
        "NOT DISPOSITIONED": 3
    }
}

df_numeric.replace(koi_disposition_labels, inplace=True)
df_numeric
# this is train data

# first we remove all string type columns from the dataframe

df_numeric = df_numeric.select_dtypes(exclude=['object']).copy()
df_test = df_numeric.copy()    # test data

# second, we manually remove some columns which are not needed as mentioned above. 
# additionally, 'koi_teq_err1' and 'koi_teq_err2' have all null values so they too need to be removed

rem_cols = ['kepid', 'koi_pdisposition', 'koi_score', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_teq_err1', 'koi_teq_err2']
df_numeric.drop(rem_cols, axis=1, inplace=True)

# this is test data
rem_cols_test = [col for col in rem_cols if col not in ['koi_pdisposition', 'koi_score']]
df_test.drop(rem_cols_test, axis=1, inplace=True)



df_numeric.head()
df_test.head()
df_numeric = df_numeric[df_numeric.isnull().sum(axis=1) == 0]
df_numeric.describe()
index = df_numeric[df_numeric.koi_fpflag_nt == df_numeric.koi_fpflag_nt.max()].index
df_numeric.drop(index, inplace=True)
df_numeric.info()
df_test = df_test[df_test.isnull().sum(axis=1) == 0]
df_test.info()
df_test = df_test[df_test.koi_disposition == 2]
df_test
df_test.to_csv('koi_test.csv')
df_numeric.to_csv('koi_numeric.csv')
df_numeric1 = df_numeric.copy()
df_numeric1.info()
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(df_numeric1.corr(), annot=True, cmap="RdYlGn", ax=ax)
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

# need to exclude the `koi_disposition` column from being standardized


df_numeric1.iloc[:, 5:] = std_scaler.fit_transform(df_numeric1.iloc[:, 5:])


# df_numeric.iloc[:, 0].to_numpy().reshape(-1, 1).shape
# df_standardized_w_labels = np.c_[df_standardized, df_numeric.iloc[:, 0].to_numpy().reshape(-1, 1)]
# df_standardized_w_labels[:3]

df_numeric1.values
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
class KeplerDataset(Dataset):
    def __init__(self, test=False):
        self.dataframe_orig = pd.read_csv(koi_cumm_path)

        if (test == False):
            self.data = df_numeric1[( df_numeric1.koi_disposition == 1 ) | ( df_numeric1.koi_disposition == 0 )].values
        else:
            self.data = df_numeric1[~(( df_numeric1.koi_disposition == 1 ) | ( df_numeric1.koi_disposition == 0 ))].values
            
        self.X_data = torch.FloatTensor(self.data[:, 1:])
        self.y_data = torch.FloatTensor(self.data[:, 0])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    
    def get_col_len(self):
        return self.X_data.shape[1]
    
kepler_df = KeplerDataset()
feature, target = kepler_df[1]
target, feature
kepler_df.get_col_len()
# splitting into training and validation set

torch.manual_seed(42)

split_ratio = .7 # 70 / 30 split

train_size = int(len(kepler_df) * split_ratio)
val_size = len(kepler_df) - train_size
train_ds, val_ds = random_split(kepler_df, [train_size, val_size])

len(train_ds), len(val_ds)
batch_size = 32

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
for features, target in train_loader:
    print(features.size(), target.size())
    break
class KOIClassifier(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(KOIClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)    
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, out_dim)
        
        
        
    def forward(self, xb):
        out = self.linear1(xb)
        out = torch.sigmoid(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        out = self.linear3(out)
        out = torch.sigmoid(out)
        out = self.linear4(out)
        out = torch.sigmoid(out)
        out = self.linear5(out)
        out = torch.sigmoid(out)

    
        return out
    
    
    def predict(self, x):
        pred = self.forward(x)
        return pred
    
        
    def print_params(self):
        for params in self.parameters():
            print(params)

input_dim = kepler_df.get_col_len()
out_dim = 1
model = KOIClassifier(input_dim, out_dim)



"""

model_prev = KOIClassifier(input_dim, out_dim)
construct = torch.load('../input/first-nn-stats/checkpoint.pth')
model_prev.load_state_dict(construct['state_dict'])

import seaborn as sns
%matplotlib inline

cf_mat_train = pred_confusion_matrix(model_prev, train_loader)
cf_mat_val = pred_confusion_matrix(model_prev, val_loader)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

ax1, ax2 = axes
sns.heatmap(cf_mat_train, fmt='g', annot=True, ax=ax1)
ax1.set_title('Training Data')

sns.heatmap(cf_mat_val, fmt='g', annot=True, ax=ax2)
ax2.set_title('Validation Data')

"""
# training phase
criterion = nn.BCELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)
n_epochs = 1000

def train_model():
    for X, y in train_loader:
        for epoch in range(n_epochs):
            optim.zero_grad()
            y_pred = model.forward(X).flatten()
            loss = criterion(y_pred, y)
            loss.backward()
            optim.step()

train_model()
# testing the predictions
for X, y in train_loader:
    y_pred = model.forward(X)
    y_pred = y_pred > 0.5
    y_pred = torch.tensor(y_pred, dtype=torch.int32)
    print(y_pred)
    break
from sklearn.metrics import confusion_matrix
def pred_confusion_matrix(model, loader):
    with torch.no_grad():
        all_preds = torch.tensor([])
        all_true = torch.tensor([])
        for X, y in loader:
            y_pred = model(X)
            y_pred = torch.tensor(y_pred > 0.5, dtype=torch.float32).flatten()
            all_preds = torch.cat([all_preds, y_pred])

            all_true = torch.cat([all_true, y])
            
    
    return confusion_matrix(all_true.numpy(), all_preds.numpy())
import seaborn as sns
%matplotlib inline

cf_mat_train = pred_confusion_matrix(model, train_loader)
cf_mat_val = pred_confusion_matrix(model, val_loader)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

ax1, ax2 = axes
sns.heatmap(cf_mat_train, fmt='g', annot=True, ax=ax1)
ax1.set_title('Training Data')

sns.heatmap(cf_mat_val, fmt='g', annot=True, ax=ax2)
ax2.set_title('Validation Data')
checkpoint = {
    'state_dict': model.state_dict(),
    'optimizer': optim.state_dict()
}

torch.save(checkpoint, 'checkpoint.pth')
# this is where we return back to the point from where we branched, we take the numeric dataframe again and apply some feature selection
df_new = pd.read_csv('koi_numeric.csv', index_col=0)
df_new.head()
# a function to remove high correlation columns by selecting the upper triangle of the correlation matrix
# and dropping all columns which have corr value > threshold at any row

def remove_high_corr(df, threshold):
    corr_mat = df.corr()
    trimask = corr_mat.abs().mask(~np.triu(np.ones(corr_mat.shape, dtype=bool), k=1))
    blocklist = [col for col in trimask.columns if (trimask[col] > threshold).any()]
    df.drop(columns=blocklist, axis=1,inplace=True)
    return blocklist
remove_high_corr(df_new, 0.80)
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df_new.corr(), cmap="Blues", ax=ax)
df_new.head()
df_new.to_csv('koi_numeric_reduced.csv')
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
    
device = get_default_device()
device
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

std_scaler = StandardScaler()

dataframe = pd.read_csv('koi_numeric_reduced.csv', index_col=0)

train_data = dataframe.query('not koi_disposition == 2').values

X = train_data[:, 1:]
y = train_data[:, 0]

val_size = .3
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=val_size, shuffle=True)

train_X[:, 4:] = std_scaler.fit_transform(train_X[:, 4:])
val_X[:, 4:] = std_scaler.fit_transform(val_X[:, 4:])


# print(f'train_X = {train_X.shape}\n\nval_X = {val_X.shape}\n')

class KOIDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.FloatTensor(X_data)
        self.y_data = torch.FloatTensor(y_data)
        
    
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    


    

train_ds = KOIDataset(train_X, train_y)
val_ds = KOIDataset(val_X, val_y)

for feature, target in train_ds:
    print(feature, target)
    break
    

batch_size = 64
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

for features, target in train_loader:
    print(target, features)
    break
# a function to measure prediction accuracy 

def accuracy(outputs, labels):
    output_labels = torch.round(torch.sigmoid(outputs))    # manually have to activate sigmoid since the nn does not incorporate sigmoid at final layer
    
    return torch.tensor(torch.sum(output_labels == labels.unsqueeze(1)).item() / len(output_labels))
    
from collections import OrderedDict

input_dim = train_X.shape[1]

class KOIClassifierSeq(nn.Module):
    def __init__(self):
        super(KOIClassifierSeq, self).__init__()
        self.model = nn.Sequential(OrderedDict([
              ('fc1', nn.Linear(input_dim, 24)),
              ('sigmoid1', nn.Sigmoid()),
              ('batchnorm1', nn.BatchNorm1d(24)),
              ('fc2', nn.Linear(24, 16)),
              ('sigmoid2', nn.Sigmoid()),
              ('batchnorm2', nn.BatchNorm1d(16)),
              ('dropout', nn.Dropout(p=0.1)),
              ('fc3', nn.Linear(16, 1))
            ]))
    
    def forward(self, xb):
        return self.model(xb)
    
    def training_step(self, batch):
        features, label = batch 
        out = self(features)
        loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(1)) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        features, label = batch 
        out = self(features)                    
        loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(1))   # Calculate loss
        acc = accuracy(out, label)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
@torch.no_grad()
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
model1 = to_device(KOIClassifierSeq(), device)
model1
num_epochs = 10
lr = 1e-4
history = fit(num_epochs, lr, model1, train_loader, val_loader, opt_func=torch.optim.Adam)
num_epochs = 5
lr = 1e-4
history = fit(num_epochs, lr, model1, train_loader, val_loader, opt_func=torch.optim.Adam)
# a function to calculate training accuracy

def train_accuracy(model):
    train_acc = []
    for X, y in train_loader:
        out = model(X)
        train_acc.append(accuracy(out, y))

    return torch.stack(train_acc).mean().item()
train_accuracy(model1)
from sklearn.metrics import confusion_matrix
def pred_confusion_matrix(model, loader):
    with torch.no_grad():
        all_preds = to_device(torch.tensor([]), device)
        all_true = to_device(torch.tensor([]), device)
        for X, y in loader:
            y_pred = model(X)
            y_pred = torch.round(torch.sigmoid(y_pred))
            all_preds = torch.cat([all_preds, y_pred])

            all_true = torch.cat([all_true, y.unsqueeze(1)])
            
    
    return confusion_matrix(all_true.cpu().numpy(), all_preds.cpu().numpy())
cf_mat_train = pred_confusion_matrix(model1, train_loader)
cf_mat_val = pred_confusion_matrix(model1, val_loader)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

ax1, ax2 = axes
sns.heatmap(cf_mat_train, fmt='g', annot=True, ax=ax1)
ax1.set_title('Training Data')

sns.heatmap(cf_mat_val, fmt='g', annot=True, ax=ax2)
ax2.set_title('Validation Data')
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
plot_accuracies(history)
plot_losses(history)
second_model = {
    'state_dict': model1.state_dict()
}

torch.save(second_model, 'second_model.pth')

# I have uploaded the pth file to the /input directory. If needed, you can load it from there and load it into a model instance of KOIClassifierSeq
test_df = pd.read_csv('koi_test.csv', index_col=0)
test_df
cols = [
 'koi_disposition',
 'koi_pdisposition',
 'koi_period_err2',
 'koi_impact_err2',
 'koi_duration_err2',
 'koi_depth_err2',
 'koi_prad_err2',
 'koi_insol_err1',
 'koi_insol_err2',
 'koi_steff_err2',
 'koi_srad_err2']

test_df.drop(cols, axis=1, inplace=True)
test_df.head()
test_X = test_df.iloc[:, 1:].values
test_probs = test_df.iloc[:, 0].values

test_X[:, 4:] = std_scaler.fit_transform(test_X[:, 4:])

  

KOI_test = KOIDataset(test_X, test_probs)
batch_size = 64
test_loader = DataLoader(KOI_test, batch_size, num_workers=4, pin_memory=True)
test_loader = DeviceDataLoader(test_loader, device)

for X, y in test_loader:
    print(X.size(), y.size())
    break
def predict_probs(model, X):
    probs = torch.sigmoid(model(X))
    return probs
torch.set_printoptions(precision=5, threshold=5000)
with torch.no_grad():
    for X, y in test_loader:
        #print(X, y)
        preds = torch.sigmoid(model1(X))
        for pred, true in zip(preds, y.unsqueeze(1)):
            print(f'model prediction: {pred.item()}\tKOI prediction: {true.item()}')
        break
def accuracy_test(outputs, label_prob):
    output_labels = torch.round(torch.sigmoid(outputs))    
    labels = torch.round(label_prob)
    return torch.tensor(torch.sum(output_labels == labels.unsqueeze(1)).item() / len(output_labels))
    
    
def test_accuracy(model):
    test_acc = []
    with torch.no_grad():
        for X, y in test_loader:
            out = model(X)
            test_acc.append(accuracy_test(out, y))

    return torch.stack(test_acc).mean().item()
test_accuracy(model1)
torch.save(model1.state_dict(), 'final_model_53_percent.pth')
class KOIClassifierSimple(nn.Module):
    def __init__(self):
        super(KOIClassifierSimple, self).__init__()
        self.model = nn.Sequential(OrderedDict([
              ('fc1', nn.Linear(input_dim, 24)),
              ('sigmoid1', nn.Sigmoid()),
              ('fc2', nn.Linear(24, 16)),
              ('sigmoid2', nn.Sigmoid()),
              ('fc3', nn.Linear(16, 1))
            ]))
    
    def forward(self, xb):
        return self.model(xb)
    
    def training_step(self, batch):
        features, label = batch 
        out = self(features)
        loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(1)) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        features, label = batch 
        out = self(features)                    
        loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(1))   # Calculate loss
        acc = accuracy(out, label)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
model2 = to_device(KOIClassifierSimple(), device)
model2
num_epochs = 10
lr = 1e-3
history2 = fit(num_epochs, lr, model2, train_loader, val_loader, opt_func=torch.optim.Adam)
train_accuracy(model2)
plot_accuracies(history2)
plot_losses(history2)
test_accuracy(model2)