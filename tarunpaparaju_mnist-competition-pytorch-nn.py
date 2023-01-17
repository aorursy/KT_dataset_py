import os

import gc

import time

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from keras.utils import to_categorical



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import torch

import torch.nn as nn

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader
np.random.seed(27)

torch.manual_seed(27)
test_df = pd.read_csv('../input/digit-recognizer/test.csv')

train_df = pd.read_csv('../input/digit-recognizer/train.csv')
test_df.head()
train_df.head()
def to_tensor(data):

    return [torch.FloatTensor(point) for point in data]



class MNISTDataset(Dataset):

    def __init__(self, df, X_col, y_col):

        self.features = df[X_col].values/255

        self.targets = df[y_col].values.reshape((-1, 1))



    def __len__(self):

        return len(self.targets)

    

    def __getitem__(self, idx):

        return to_tensor([self.features[idx], self.targets[idx]])
y_col = "label"

test_df[y_col] = [-1]*len(test_df)



split = int(0.8*len(train_df))

valid_df = train_df[split:].reset_index(drop=True)

train_df = train_df[:split].reset_index(drop=True)
X_col = list(train_df.columns[1:])



train_set = MNISTDataset(train_df, X_col, y_col)

valid_set = MNISTDataset(valid_df, X_col, y_col)



valid_loader = DataLoader(valid_set, batch_size=1024, shuffle=True)

train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
class MLP(nn.Module):

    def __init__(self, i, u, v, o):

        super(MLP, self).__init__()

        self.relu_layer = nn.ReLU()

        self.dense_1 = nn.Linear(i, u)

        self.dense_2 = nn.Linear(u, v)

        self.dense_output = nn.Linear(v, o)

        

    def forward(self, x):

        x = self.relu_layer(self.dense_1(x))

        x = self.relu_layer(self.dense_2(x))

        logits = self.dense_output(x); return logits
device = torch.device('cuda')

network = MLP(i=784, u=20, v=15, o=10).to(device)

optimizer = Adam(params=network.parameters(), lr=0.01)
print(network)
def cel(y_true, y_pred):

    y_true = y_true.long().squeeze()

    return nn.CrossEntropyLoss()(y_pred, y_true)



def acc(y_true, y_pred):

    y_true = y_true.long().squeeze()

    y_pred = torch.argmax(y_pred, axis=1)

    return (y_true == y_pred).float().sum()/len(y_true)
start = time.time()

print("STARTING TRAINING ...\n")



train_losses, valid_losses = [], []

train_accuracies, valid_accuracies = [], []



for epoch in range(20):

    network = network.train()

    print("Epoch: {}".format(epoch + 1))

    batch_train_losses, batch_train_accuracies = [], []

    

    batch = 0

    for train_batch in train_loader:

        train_X, train_y = train_batch



        train_X = train_X.to(device)

        train_y = train_y.to(device)

        train_preds = network.forward(train_X)

        train_loss = cel(train_y, train_preds)

        train_accuracy = acc(train_y, train_preds)

        

        optimizer.zero_grad()

        train_loss.backward()

        

        optimizer.step()

        train_loss = np.round(train_loss.item(), 3)

        train_accuracy = np.round(train_accuracy.item(), 3)



        end = time.time()

        batch = batch + 1

        log = batch % 10 == 0

        time_delta = np.round(end - start, 3)

        

        batch_train_losses.append(train_loss)

        batch_train_accuracies.append(train_accuracy)

        logs = "Batch: {} || Train Loss: {} || Train Acc: {} || Time: {} s"

        if log: print(logs.format(batch, train_loss, train_accuracy, time_delta))

        

    train_losses.append(np.mean(batch_train_losses))

    train_accuracies.append(np.mean(batch_train_accuracies))

    

    total_valid_loss = 0

    total_valid_points = 0

    total_valid_accuracy = 0

    

    with torch.no_grad():

        for valid_batch in valid_loader:

            valid_X, valid_y = valid_batch

            

            valid_X = valid_X.to(device)

            valid_y = valid_y.to(device)

            valid_preds = network.forward(valid_X)

            valid_loss = cel(valid_y, valid_preds)

            valid_accuracy = acc(valid_y, valid_preds)

            

            total_valid_points += 1

            total_valid_loss += valid_loss.item()

            total_valid_accuracy += valid_accuracy.item()

            

    valid_loss = np.round(total_valid_loss/total_valid_points, 3)

    valid_accuracy = np.round(total_valid_accuracy/total_valid_points, 3)

    

    valid_losses.append(valid_loss)

    valid_accuracies.append(valid_accuracy)

    

    end = time.time()

    time_delta = np.round(end - start, 3)

    logs = "Epoch: {} || Valid Loss: {} || Valid Acc: {} || Time: {} s"

    print("\n" + logs.format(epoch + 1, valid_loss, valid_accuracy, time_delta) + "\n")

    

print("ENDING TRAINING ...")
fig = go.Figure()



fig.add_trace(go.Scatter(x=np.arange(1, len(valid_losses)),

                         y=valid_losses, mode="lines+markers", name="valid",

                         marker=dict(color="indianred", line=dict(width=.5,

                                                                  color='rgb(0, 0, 0)'))))



fig.add_trace(go.Scatter(x=np.arange(1, len(train_losses)),

                         y=train_losses, mode="lines+markers", name="train",

                         marker=dict(color="darkorange", line=dict(width=.5,

                                                                   color='rgb(0, 0, 0)'))))



fig.update_layout(xaxis_title="Epochs", yaxis_title="Cross Entropy",

                  title_text="Cross Entropy vs. Epochs", template="plotly_white", paper_bgcolor="#f0f0f0")



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=np.arange(1, len(valid_accuracies)),

                         y=valid_accuracies, mode="lines+markers", name="valid",

                         marker=dict(color="indianred", line=dict(width=.5,

                                                                  color='rgb(0, 0, 0)'))))



fig.add_trace(go.Scatter(x=np.arange(1, len(train_accuracies)),

                         y=train_accuracies, mode="lines+markers", name="train",

                         marker=dict(color="darkorange", line=dict(width=.5,

                                                                   color='rgb(0, 0, 0)'))))



fig.update_layout(xaxis_title="Epochs", yaxis_title="Accuracy",

                  title_text="Accuracy vs. Epochs", template="plotly_white", paper_bgcolor="#f0f0f0")



fig.show()
def softmax(x):

    return np.exp(x)/np.sum(np.exp(x), axis=1)[:, None]
test_set = MNISTDataset(test_df, X_col, y_col)

test_loader = tqdm(DataLoader(test_set, batch_size=1024, shuffle=False))



test_preds = []

with torch.no_grad():

    for test_X, _ in test_loader:

        test_X = test_X.to(device)

        test_pred = network.forward(test_X)

        test_preds.append(softmax(test_pred.detach().cpu().numpy()))
submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

submission["Label"] = np.argmax(np.concatenate(test_preds, axis=0), axis=1)
test_batch = next(iter(test_loader))[0]

test_X = test_batch.reshape(-1, 28, 28)[:36]

fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(15, 15))



for i, image in enumerate(test_X):

    ax[i//6][i%6].axis('off'); ax[i//6][i%6].imshow(image, cmap='gray')

    ax[i//6][i%6].set_title(np.argmax(test_preds[0][i], axis=0), fontsize=20, color="red")
submission.head(10)
submission.to_csv("submission.csv", index=False)