!pip install jovian --quiet --upgrade
# imports
import jovian
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
df = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df.head() #overview of data format
df.shape
df.columns # shows all columns in dataset ; len(df.columns) = 40
df.drop(['gameId','redFirstBlood','blueTotalGold','redTotalGold','blueTotalExperience','redTotalExperience','redGoldDiff','redExperienceDiff','redKills','redDeaths'], axis=1, inplace=True)
df.head()
targets = df[['blueWins']].values
features = df.drop('blueWins', axis=1).values
features.shape , targets.shape
test_size = int(.10 * 9879) # represents size of validation set
val_size = test_size
train_size = 9879 - test_size*2
train_size , val_size, test_size
dataset = TensorDataset(torch.tensor(features).float(), torch.from_numpy(targets).float())
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size)
for xb, yb in train_loader:
    print(xb.shape, yb.shape)
    print(xb, yb)
    break # check if train_loader works
input_size = 29 # all the feature columns
output_size = 1 # probability of 0 to 1 in the chances of blue side winning
threshold = 0.5
# here the 'fun' begins!
class LOLModel(nn.Module):
    def __init__(self):
        # initiate the model
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, xb):
        # forward function of the model 
        out = self.sigmoid(self.linear(xb))
        return out
    
    def training_step(self, batch):
        # used for training per batch in an eopch
        inputs, labels = batch
        out = self(inputs)
        loss = F.binary_cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        # used on function `evaluate` to iterate model through a batch
        inputs, labels = batch
        out = self(inputs)
        loss = F.binary_cross_entropy(out, labels)
        acc = accuracy(out, labels)
        # `.detach()` makes sure gradient is not tracked
        return {'val_loss': loss.detach(), 'val_acc' : acc.detach()}
    
    def validation_epoch_end(self, outputs):
        # calculate mean loss and accuracy for batch called w/ `evaluate`
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # print function to see what's going on
        if ((epoch+1) % 10 == 0) or (epoch == (num_epochs-1)):
            # print for every 5 epochs
            print("Epoch [{}], val_loss: {:.4f}, val_acc {:.4f}".format(epoch+1, result['val_loss'], result['val_acc']))
def accuracy(out, labels):
    return torch.tensor(torch.sum(abs(out-labels) < threshold).item() / len(out))
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history
# Setup for train loop
model = LOLModel()
# model.double()
evaluate(model, val_loader)
history = fit(750, .0001, model, train_loader, val_loader)
accuracies = [r['val_acc'] for r in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. No. of epochs')
evaluate(model, test_loader)
torch.save(model.state_dict(), 'lol_logistic.pth')
jovian.log_dataset(path='/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv', description='LOL diamond-level ingame data within the first 10 minutes')
jovian.log_metrics(val_acc = history[-1]['val_acc'], val_loss = history[-1]['val_loss'])
jovian.log_hyperparams(lr = 0.0001, epochs = 750, batch_size = 128)
jovian.commit(project='lol-logistic',environment=None, outputs=['lol_logistic.pth'])
