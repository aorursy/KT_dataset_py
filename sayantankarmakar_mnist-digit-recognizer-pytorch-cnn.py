import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm



%matplotlib inline
import torch

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
np.random.seed(1)

torch.manual_seed(1)
train_csv = '../input/digit-recognizer/train.csv'

test_csv = '../input/digit-recognizer/test.csv'



train_df = pd.read_csv(train_csv)

test_df = pd.read_csv(test_csv)
test_df["label"] = [-1]*len(test_df)
train_df.head()
test_df.head()
from sklearn.model_selection import KFold
train_df['fold'] = -1



FOLDS = 5



kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1)



for i, (_, val_idx) in enumerate(kf.split(train_df)):

    train_df.loc[val_idx, 'fold'] = i



train_df.head(3)
def to_tensor(data):

    # Utility function to convert data into PyTorch tensors

    return [torch.FloatTensor(point) for point in data]



class MNISTDataset(Dataset):

    def __init__(self, df, X_col, y_col):

        """

            df: the pandas DataFrame to be referred

            X_col: the columns representing the pixel values of the image.

            y_col: the column representing the target label.

        """

        self.features = (df[X_col].values/255).reshape((-1,1,28,28)) # Reshaping images to form (batch_size, channels, height, width)

        self.targets = df[y_col].values.reshape((-1,1)) 

        

    def __len__(self):

        return len(self.targets)

    

    def __getitem__(self, idx):

        return to_tensor([self.features[idx], self.targets[idx]])
class ConvNet(torch.nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()

        self.relu_layer = torch.nn.ReLU()

        self.conv1 = torch.nn.Conv2d(1, 32, 5)

        self.conv2 = torch.nn.Conv2d(32, 64, 5)

        self.max = torch.nn.MaxPool2d(2)

        self.fc1 = torch.nn.Linear(4*4*64, 256)

        self.fc2 = torch.nn.Linear(256, 128)

        self.fc3 = torch.nn.Linear(128, 10)

    

    def forward(self, x, train=True):

        x = self.conv1(x)

        x = self.relu_layer(x)

        x = self.max(x) # First Conv Block

        

        x = self.conv2(x)

        x = self.relu_layer(x)

        x = self.max(x) # Second Conv Block

        

        x = x.view(-1, 4*4*64) # Flattening

        

        x = self.fc1(x)

        x = self.relu_layer(x)

                

        x = self.fc2(x)

        x = self.relu_layer(x)

        

        x = self.fc3(x)

        

        if train:

            return x

        else:

            return F.log_softmax(x, dim=1)
def cel(y_pred, y_true):

    y_true = y_true.long().squeeze()

    return torch.nn.CrossEntropyLoss()(y_pred, y_true)



def acc(y_pred, y_true):

    y_true = y_true.long().squeeze()

    y_pred = torch.argmax(y_pred, axis=1)

    return (y_true == y_pred).float().sum()/len(y_true)
EPOCHS = 10



all_train_losses = np.zeros((EPOCHS))

all_valid_losses = np.zeros((EPOCHS))



all_train_accs = np.zeros((EPOCHS))

all_valid_accs = np.zeros((EPOCHS))
def train(fold):

    print("Starting Training for FOLD {} ....".format(fold))

    

    train_bs = 64

    valid_bs = 32

    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    train_fold = train_df[train_df.fold != fold].reset_index(drop=True)

    valid_fold = train_df[train_df.fold == fold].reset_index(drop=True)

    

    model = ConvNet()

    model.to(device)

    

    X_col = list(train_df.columns[1:-1])

    y_col = "label"

        

    train_set = MNISTDataset(train_fold, X_col, y_col)

    valid_set = MNISTDataset(valid_fold, X_col, y_col)

    

    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True)

    valid_loader = DataLoader(valid_set, batch_size=valid_bs, shuffle=True)

    

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)

    

    train_losses, valid_losses = [], []

    train_accs, valid_accs = [], []

    

    for epoch in range(EPOCHS):

        model = model.train()

        print("Epoch: {}".format(epoch+1))

        

        batch = 0

        batch_train_losses, batch_train_accs = [], []

        for train_batch in train_loader:

            train_X, train_y = train_batch

            

            train_X = train_X.to(device)

            train_y = train_y.to(device)

            

            train_preds = model.forward(train_X, train=True)

            train_loss = cel(train_preds, train_y)

            train_acc = acc(train_preds, train_y)

            

            optimizer.zero_grad()

            train_loss.backward()

            

            optimizer.step()

            train_loss = np.round(train_loss.item(), 3)

            train_acc = np.round(train_acc.item(), 3)

            

            batch += 1

            log = batch % 10 == 0

            

            batch_train_losses.append(train_loss)

            batch_train_accs.append(train_acc)

            if log:

                print(">", end="")

        

        logs = "\nTrain Loss: {} || Train Acc: {}"

        print(logs.format(np.round(np.mean(batch_train_losses), 3), np.round(np.mean(batch_train_accs), 3)))

        

        train_losses.append(np.mean(batch_train_losses))

        train_accs.append(np.mean(batch_train_accs))

        

        total_valid_points = 0

        total_valid_loss = 0

        total_valid_acc = 0

        

        with torch.no_grad():

            for valid_batch in valid_loader:

                valid_X, valid_y = valid_batch

                

                valid_X = valid_X.to(device)

                valid_y = valid_y.to(device)

                

                valid_preds = model.forward(valid_X, train=True) # To use Loss function, we need raw outputs, hence train=True

                valid_loss = cel(valid_preds, valid_y)

                valid_acc = acc(valid_preds, valid_y)

                

                total_valid_points += 1

                total_valid_loss += valid_loss.item()

                total_valid_acc += valid_acc.item()

        

        valid_loss = np.round(total_valid_loss / total_valid_points, 3)

        valid_acc = np.round(total_valid_acc / total_valid_points, 3)

        

        valid_losses.append(valid_loss)

        valid_accs.append(valid_acc)

        

        logs = "Valid Loss: {} || Valid Acc: {}"

        print(logs.format(valid_loss, valid_acc) + "\n")

    

    MODEL_PATH = "model_fold_{}.pt"

    torch.save(model.state_dict(), MODEL_PATH.format(fold))

    

    print("Ending Training for FOLD {}.... \n".format(fold))

    

    return train_losses, valid_losses, train_accs, valid_accs
for fold in range(FOLDS):

    train_losses, valid_losses, train_accs, valid_accs = train(fold)

    

    all_train_losses += train_losses

    all_valid_losses += valid_losses

    

    all_train_accs += train_accs

    all_valid_accs += valid_accs
all_train_losses /= FOLDS

all_valid_losses /= FOLDS



all_train_accs /= FOLDS

all_valid_accs /= FOLDS
def modify(train_vals, valid_vals):

    train_hue = np.zeros(len(train_vals)).reshape(-1,1)

    valid_hue = np.ones(len(valid_vals)).reshape(-1,1)

    

    epo = np.arange(1, EPOCHS+1).reshape(-1,1)



    train_vals = np.round(train_vals.reshape(-1,1), 3)

    valid_vals = np.round(valid_vals.reshape(-1,1), 3)

    

    train = np.concatenate((train_vals, train_hue, epo), axis=1)

    valid = np.concatenate((valid_vals, valid_hue, epo), axis=1)

    

    return np.concatenate((train, valid), axis=0)
losses = modify(all_train_losses, all_valid_losses)

accs = modify(all_train_accs, all_valid_accs)
loss_df = pd.DataFrame(data=losses, columns=['loss', 'is_valid', 'epoch'])

acc_df = pd.DataFrame(data=accs, columns=['acc', 'is_valid', 'epoch'])



loss_df.to_csv('loss.csv', index=False)

acc_df.to_csv('acc.csv', index=False)
plt.figure(figsize=(16,9))

sns.lineplot(x='epoch', y='loss', data=loss_df, hue='is_valid', style='is_valid')

plt.show()
plt.figure(figsize=(16,9))

sns.lineplot(x='epoch', y='acc', hue='is_valid', data=acc_df, style='is_valid')

plt.show()
def predict(fold):

    

    bs = 64

    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    MODEL_PATH = "model_fold_{}.pt".format(fold)

    

    model = ConvNet()

    model.load_state_dict(torch.load(MODEL_PATH))

    model.to(device)

    

    X_col = list(test_df.columns[:-1])

    y_col = "label"

    

    test_set = MNISTDataset(test_df, X_col, y_col)

    test_loader = tqdm(DataLoader(test_set, batch_size=bs, shuffle=False))

    

    test_preds = []

    

    with torch.no_grad():

        for test_X, _ in test_loader:

            test_X = test_X.to(device)

            batch_test_preds = model.forward(test_X, train=False) # Outputs softmax values

            test_preds.append(batch_test_preds.cpu().detach().numpy())

            

    test_preds = np.concatenate(test_preds, axis=0)

    

    return test_preds
preds = np.empty(shape=(FOLDS, test_df.shape[0], 10))
for fold in range(FOLDS):

    preds[fold] = predict(fold)
subs = np.zeros((test_df.shape[0], 10))
for fold in range(FOLDS):

    subs += preds[fold]



subs /= 5
X_col = list(test_df.columns[:-1])

y_col = "label"

    

test_set = MNISTDataset(test_df, X_col, y_col)

test_loader = tqdm(DataLoader(test_set, batch_size=12, shuffle=False))



test_batch = next(iter(test_loader))[0]

test_X = test_batch.reshape(-1, 28, 28)[:12]

fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(12, 6))



for i, image in enumerate(test_X):

    ax[i//6][i%6].axis('off'); ax[i//6][i%6].imshow(image, cmap='gray')

    ax[i//6][i%6].set_title(np.argmax(subs[i], axis=0), fontsize=20, color="red")
submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

submission["Label"] = np.argmax(subs, axis=1)



submission.head()
submission.to_csv("submission.csv", index=False)