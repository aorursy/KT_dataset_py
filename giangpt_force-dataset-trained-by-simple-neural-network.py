import pandas as pd

import numpy as np

from collections import Counter

from sklearn.preprocessing import LabelEncoder

import torch

from torch.utils.data import Dataset, DataLoader

import torch.optim as torch_optim

import torch.nn as nn

import torch.nn.functional as F

from torchvision import models

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix

import seaborn as sns
input_data = "../input/forcedataset/force-ai-well-logs/train.csv"
TARGET_1 = "FORCE_2020_LITHOFACIES_LITHOLOGY"

TARGET_2 = "FORCE_2020_LITHOFACIES_CONFIDENCE"

WELL_NAME = 'WELL'
#read csv data

df = pd.read_csv(input_data, sep=';', nrows = 200000)
wells = np.unique(df['WELL'].values)
len(wells)
#number: rock types dictionary

lithology_keys = {30000: 'Sandstone',

                 65030: 'Sandstone/Shale',

                 65000: 'Shale',

                 80000: 'Marl',

                 74000: 'Dolomite',

                 70000: 'Limestone',

                 70032: 'Chalk',

                 88000: 'Halite',

                 86000: 'Anhydrite',

                 99000: 'Tuff',

                 90000: 'Coal',

                 93000: 'Basement'}
lithology_map = {30000: 0,

                65030: 1,

                65000: 2, 

                80000: 3,

               74000: 4,

               70000: 5,

                70032: 6,

                88000: 7,

                86000: 8,

               99000: 9,

               90000: 10,

               93000: 11}
df['LITHOLOGY'] = df[TARGET_1].map(lithology_map)
rock_lst = list(lithology_map.keys())
rock_lst
rock_types = np.unique(df['FORCE_2020_LITHOFACIES_LITHOLOGY'].values)
number_of_rockTypes = len(rock_types)
number_of_rockTypes
unused_columns = ['RSHA', 'SGR', 'NPHI', 'BS', 'DTS', 'DCAL', 'RMIC', 'ROPA', 'RXO']

unused_columns += [WELL_NAME, 'GROUP', 'FORMATION']

# ADD two target columns into unused columns

unused_columns += [TARGET_1, TARGET_2, 'LITHOLOGY']
all_columns = list(df.columns)



use_columns = [c for c in all_columns if c not in unused_columns]
for c in use_columns:

    df[c].fillna(df[c].mean(), inplace=True)

train_wells = list(np.unique(df['WELL'].values))[:int(len(wells) * 0.8)]

# Use this condition to find out which rows in the data is select for training

train_mask = df[WELL_NAME].isin(train_wells)
X_train = df[train_mask][use_columns].values

y_train = df[train_mask]['LITHOLOGY'].values

print(X_train.shape, y_train.shape)
X_valid = df[~train_mask][use_columns].values

y_valid = df[~train_mask]['LITHOLOGY'].values

print(X_valid.shape, y_valid.shape)
def get_default_device():

    """Pick GPU if available, else CPU"""

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
class ForceDataset(Dataset):

    def __init__(self, X, y):

        X = X.copy()

        self.X = torch.from_numpy(X).float().to(device)

        self.y = torch.from_numpy(y).long().to(device)

        

    def __len__(self):

        return len(self.y)

    

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]
train_ds = ForceDataset(X_train, y_train)

valid_ds = ForceDataset(X_valid, y_valid)
class ForceModel(nn.Module):

    def __init__(self, n_cont):

        super().__init__()

        self.n_cont = n_cont

        self.lin1 = nn.Linear(self.n_cont, 200)

        self.lin2 = nn.Linear(200, 70)

        self.lin3 = nn.Linear(70, number_of_rockTypes)

        self.bn1 = nn.BatchNorm1d(self.n_cont)

        self.bn2 = nn.BatchNorm1d(200)

        self.bn3 = nn.BatchNorm1d(70)

        self.drops = nn.Dropout(0.3)

        



    def forward(self, x_cont):

        x = self.bn1(x_cont)

        x = F.relu(self.lin1(x))

        x = self.drops(x)

        x = self.bn2(x)

        x = F.relu(self.lin2(x))

        x = self.drops(x)

        x = self.bn3(x)

        x = self.lin3(x)

        return x
model = ForceModel(len(use_columns))

to_device(model, device)
def get_optimizer(model, lr = 0.001, wd = 0.0):

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)

    return optim
def train_model(model, optim, train_dl):

    model.train()

    total = 0

    sum_loss = 0

    for x, y in train_dl:

        batch = y.shape[0]

        output = model(x)

        loss = F.cross_entropy(output, y)   

        optim.zero_grad()

        loss.backward()

        optim.step()

        total += batch

        sum_loss += batch*(loss.item())

    return sum_loss/total
def val_loss(model, valid_dl):

    model.eval()

    total = 0

    sum_loss = 0

    correct = 0

    for x , y in valid_dl:

        current_batch_size = y.shape[0]

        out = model(x)

        loss = F.cross_entropy(out, y)

        sum_loss += current_batch_size*(loss.item())

        total += current_batch_size

        pred = torch.max(out, 1)[1]

        correct += (pred == y).float().sum().item()

    print("valid loss %.3f and accuracy %.3f" % (sum_loss/total, correct/total))

    return sum_loss/total, correct/total

def train_loop(model, epochs, lr=0.01, wd=0.0):

    optim = get_optimizer(model, lr = lr, wd = wd)

    for i in range(epochs): 

        loss = train_model(model, optim, train_dl)

        print("training loss: ", loss)

        

        val_loss(model, valid_dl)
batch_size = 1000

train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)

valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)
train_loop(model, epochs= 100, lr=0.01, wd=0.00001)
def get_prediction(model, valid_ds):

    out = model(valid_ds.X)

    final_pred = np.array([item.argmax() for item in out])

    return final_pred
# plot confusion matrix

def draw_confusion_matrix(y_true, y_pred):

    cfs = np.zeros((number_of_rockTypes, number_of_rockTypes));

    for i in range(len(y_true)):

        cfs[y_true[i]][y_pred[i]] += 1;



    for i in range(number_of_rockTypes):

        total_row = 0

        for j in range(number_of_rockTypes):

            total_row += cfs[i][j]

        if total_row != 0:

            for j in range(number_of_rockTypes):

                cfs[i][j] /= total_row

        

            

    plt.figure(figsize=(20,10))

    plt.title("Confusion matrix")

    sns.heatmap(cfs, annot = True, xticklabels = rock_lst, yticklabels = rock_lst)

    plt.show()
# accuracy calculation

def calculate_accuracy(y_true, y_pred):

    cm = np.zeros((number_of_rockTypes, number_of_rockTypes));

    for i in range(len(y_true)):

        cm[y_true[i]][y_pred[i]] += 1;

    tp = 0

    for i in range(len(cm)):

        tp += cm[i][i]

    accuracy = 1.0 * tp / np.sum(cm)

    return accuracy

y_valid_pred = get_prediction(model, valid_ds)
draw_confusion_matrix(y_valid, y_valid_pred)
calculate_accuracy(y_valid, y_valid_pred)
penalty_matrix = np.load("../input/penalty-matrix/penalty_matrix.npy")
# Position of each type of rock in the penalty_matrix

penalty_dict = {"Sandstone": 0,

                "Sandstone/Shale": 1,

                "Shale": 2, 

                "Marl": 3,

                "Dolomite": 4,

                "Limestone": 5,

                "Chalk": 6,

                "Halite": 7,

                "Anhydrite": 8,

                "Tuff": 9,

                "Coal": 10,

                "Basement": 11}
cfs = np.zeros((number_of_rockTypes, number_of_rockTypes));

for i in range(len(y_valid)):

    cfs[y_valid[i]][y_valid_pred[i]] += 1;
# penalty calculation according to FORCE metrics.

def calculate_penalty(cm = None, penalty_matrix = None, lithology_dict = None, penalty_dict = None, cm_rock_idx = None):

    sum_penalty = 0

    for i in range(len(cm)):

        for j in range(len(cm)):

            rock_i = lithology_dict[cm_rock_idx[i]]

            rock_j = lithology_dict[cm_rock_idx[j]]

            penalty_i = penalty_dict[rock_i]

            penalty_j = penalty_dict[rock_j]

            sum_penalty += cm[i][j] * penalty_matrix[penalty_i][penalty_j]

    return -1.0 * sum_penalty / np.sum(cm)
# Used for getting the right "rock number" from confusion matrix index

cm_rock_idx = np.unique(df[TARGET_1].values)