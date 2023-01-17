TRAIN_PATH = '/kaggle/input/deepfake-detection-data-preparation/train.csv'

SAVE_PATH = '/kaggle/working/model.pth'



TEST_SIZE = 0.3

RANDOM_STATE = 128

EPOCHS = 200

BATCH_SIZE = 64

LR = 1e-4
import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')
class LogisticRegression(nn.Module):

    def __init__(self, D_in=1, D_out=1):

        super(LogisticRegression, self).__init__()

        self.linear = nn.Linear(D_in, D_out)

        

    def forward(self, x):

        y_pred = self.linear(x)



        return y_pred

    

    def predict(self, x):

        result = self.forward(x)



        return torch.sigmoid(result)
def shuffle_data(X, y):

    assert len(X) == len(y)

    

    p = np.random.permutation(len(X))

    

    return X[p], y[p]
train_df = pd.read_csv(TRAIN_PATH)

train_df.head()
label_count = train_df.groupby('label').count()['filename']

print(label_count)



# Use pos_weight value to overcome imbalanced dataset.

# https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss

pos_weight = torch.ones([1]) * label_count[0]/label_count[1]

print('pos_weight:', pos_weight)
X = train_df['distance'].to_numpy()

y = train_df['label'].to_numpy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_train = torch.tensor(X_train).to(device).unsqueeze(dim=1).float()

X_val = torch.tensor(X_val).to(device).unsqueeze(dim=1).float()

y_train = torch.tensor(y_train).to(device).unsqueeze(dim=1).float()

y_val = torch.tensor(y_val).to(device).unsqueeze(dim=1).float()
classifier = LogisticRegression()

criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight) # Improve stability

optimizer = optim.Adam(classifier.parameters(), lr=LR)



n_batches = np.ceil(len(X_train) / BATCH_SIZE).astype(int)

losses = np.zeros(EPOCHS)

val_losses = np.zeros(EPOCHS)

best_val_loss = 1e7



for e in tqdm(range(EPOCHS)):

    batch_losses = np.zeros(n_batches)

    pbar = tqdm(range(n_batches))

    pbar.desc = f'Epoch {e+1}'

    classifier.train()

    

    # Shuffle training data

    X_train, y_train = shuffle_data(X_train, y_train)



    for i in pbar:

        # Get batch.

        X_batch = X_train[i*BATCH_SIZE:min(len(X_train), (i+1)*BATCH_SIZE)]

        y_batch = y_train[i*BATCH_SIZE:min(len(y_train), (i+1)*BATCH_SIZE)]



        # Make prediction.

        y_pred = classifier(X_batch)



        # Compute loss.

        loss = criterion(y_pred, y_batch)

        batch_losses[i] = loss



        # Zero gradients, perform a backward pass, and update the weights.

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    

    # Compute batch loss (average)

    losses[e] = batch_losses.mean()

    

    # Compute val loss

    classifier.eval()

    y_val_pred = classifier(X_val)

    val_losses[e] = criterion(y_val_pred, y_val)

    

    # Save model based on the best (lowest) val loss.

    if val_losses[e] < best_val_loss:

        print('Found a better checkpoint!')

        torch.save(classifier.state_dict(), SAVE_PATH)

        best_val_loss = val_losses[e]

        

    

    # Display some information in progress-bar.

    pbar.set_postfix({

        'loss': losses[e],

        'val_loss': val_losses[e]

    })
fig = plt.figure(figsize=(16, 8))

ax = fig.add_axes([0, 0, 1, 1])



ax.plot(np.arange(EPOCHS), losses)

ax.plot(np.arange(EPOCHS), val_losses)

ax.set_xlabel('epoch', fontsize='xx-large')

ax.set_ylabel('log loss', fontsize='xx-large')

ax.legend(

    ['loss', 'val loss'],

    loc='upper right',

    fontsize='xx-large',

    shadow=True

)

plt.show()
without_weight_criterion = nn.BCELoss(reduction='mean')



classifier.eval()

with torch.no_grad():

    y_val_pred = classifier.predict(X_val)

    val_loss = without_weight_criterion(y_val_pred, y_val)



print('val loss:', val_loss.detach().numpy())
plt.hist(y_val_pred.squeeze(dim=-1).detach())

plt.plot()