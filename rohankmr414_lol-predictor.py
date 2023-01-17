import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from tqdm.notebook import tqdm
df = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df.head()
df.columns
df.drop(['gameId','redFirstBlood','blueTotalGold','redTotalGold','blueTotalExperience','redTotalExperience','redGoldDiff','redExperienceDiff','redKills','redDeaths'], axis=1, inplace=True)
df.head()
y = df[['blueWins']].values

X = df.drop('blueWins', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
BATCH_SIZE = 512

LEARNING_RATE = 0.01

EPOCHS = 100
## train data

class trainData(Dataset):

    

    def __init__(self, X_data, y_data):

        self.X_data = X_data

        self.y_data = y_data

        

    def __getitem__(self, index):

        return self.X_data[index], self.y_data[index]

        

    def __len__ (self):

        return len(self.X_data)





train_data = trainData(torch.FloatTensor(X_train), 

                       torch.FloatTensor(y_train))

## test data    

class testData(Dataset):

    

    def __init__(self, X_data):

        self.X_data = X_data

        

    def __getitem__(self, index):

        return self.X_data[index]

        

    def __len__ (self):

        return len(self.X_data)

    



test_data = testData(torch.FloatTensor(X_test))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(dataset=test_data, batch_size=1)
class LOLClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer_1 = nn.Linear(29, 64)

        self.layer_2 = nn.Linear(64, 32)

        self.layer_3 = nn.Linear(32, 16)

        self.layer_4 = nn.Linear(16, 8)

        self.layer_out = nn.Linear(8, 1)

        

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        

        self.dropout1 = nn.Dropout(p=0.2)

        self.dropout = nn.Dropout(p=0.1)

        self.batchnorm1 = nn.BatchNorm1d(29)

#         self.batchnorm2 = nn.BatchNorm1d(64)

#         self.batchnorm3 = nn.BatchNorm1d(16)

#         self.batchnorm4 = nn.BatchNorm1d(8)



        

    def forward(self, inputs):

        x = self.batchnorm1(inputs)

        x = self.relu(self.layer_1(x))

#         x = self.batchnorm2(x)

        x = self.relu(self.layer_2(x))

#         x = self.dropout(x)

#         x = self.batchnorm2(x)

        x = self.relu(self.layer_3(x))

#         x = self.batchnorm3(x)

#         x = self.dropout2(x)

        x = self.relu(self.layer_4(x))

#         x = self.batchnorm4(x)

        x = self.sigmoid(self.layer_out(x))

        return x

        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
model = LOLClassifier()

model.to(device)



criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
def binary_acc(y_pred, y_test):

    y_pred_tag = torch.round(torch.sigmoid(y_pred))



    correct_results_sum = (y_pred_tag == y_test).sum().float()

    acc = correct_results_sum/y_test.shape[0]

    acc = torch.round(acc * 100)

    

    return acc
losses = []

accs = []

epchs = []



model.train()

for e in range(1, EPOCHS+1):

    epoch_loss = 0

    epoch_acc = 0

    for X_batch, y_batch in train_loader:

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        

        y_pred = model(X_batch)

        

        loss = criterion(y_pred, y_batch)

        acc = binary_acc(y_pred, y_batch)

        

        loss.backward()

        optimizer.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

    

    losses.append(epoch_loss/len(train_loader))

    accs.append(epoch_acc/len(train_loader))

    epchs.append(e)

    

#     scheduler.step()



    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
y_pred_list = []

model.eval()

with torch.no_grad():

    for X_batch in test_loader:

        X_batch = X_batch.to(device)

        y_test_pred = model(X_batch)

        y_test_pred = torch.sigmoid(y_test_pred)

        y_pred_tag = torch.round(y_test_pred)

        y_pred_list.append(y_pred_tag.cpu().numpy())



y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
confusion_matrix(y_test, y_pred_list)
print(classification_report(y_test, y_pred_list))
# accuracies = [r['val_loss'] for r in history]

plt.plot(losses, '-x')

plt.xlabel('epoch')

plt.ylabel('Loss')

plt.title('Loss vs. No. of epochs')

plt.show()



plt.plot(accs, '-x')

plt.xlabel('epoch')

plt.ylabel('Accuracy')

plt.title('Accuracy vs. No. of epochs')

plt.show()