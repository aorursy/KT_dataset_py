import torch
import torchvision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
%matplotlib inline
train = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")
test = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test.csv")
train.head()
labels = train['label']
train.drop('label', axis = 1, inplace = True)
test_labels = test['label']
test.drop('label', axis = 1, inplace = True)

train_data = train.values
labels = labels.values
test_data = test.values
test_labels = test_labels.values
alphabets = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:"f", 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n',
        14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'}
plt.figure(figsize = (18, 18))

figure_data = train_data[10].reshape(28, 28)
plt.subplot(221)
plt.title("Alphabet: {}".format(alphabets[int(labels[10])]))
print("Alphabet:", (alphabets[int(labels[10])]))
sns.heatmap(data = figure_data)

figure_data = train_data[11].reshape(28, 28)
plt.subplot(222)
plt.title("Alphabet: {}".format(alphabets[int(labels[11])]))
print("Alphabet:", (alphabets[int(labels[11])]))
sns.heatmap(data = figure_data)
from collections import Counter
counter = Counter(labels)
counter
train_size = int(len(train) - 0.2*len(train))
val_size = int(0.2*len(train))
type(train_data)
type(labels)
# new = []
# for ind, i in enumerate(train_data):
#      new.append(train_data[ind].astype('float64') / 255.0)

# train_data = new
train_data[0]
# new = []
# for ind, i in enumerate(test_data):
#      new.append(test_data[ind].astype('float64') / 255.0)

# test_data = new
test_data[0]
result = []

for ind, training_arr in enumerate(train_data):
    result.append(np.reshape(training_arr, (28, 28)))

result
labels
from torch.utils.data import TensorDataset
import numpy as np
labels = torch.from_numpy(labels)
temp = torch.tensor(result)

temp, labels = temp.type(dtype=torch.float32), labels.type(dtype=torch.int64)

dataset = TensorDataset(temp, labels)
test_result = []

for ind, testing_arr in enumerate(test_data):
    test_result.append(np.reshape(testing_arr, (28, 28)))

test_result
test_labels = torch.from_numpy(test_labels)
test_temp = torch.tensor(test_result)

test_temp, test_labels = test_temp.type(dtype=torch.float32), test_labels.type(dtype=torch.int64)

test = TensorDataset(test_temp, test_labels)
test[0]
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)
batch_size=128
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test, batch_size*2, num_workers=4, pin_memory=True)
for x in train_loader:
    print(x)
    break
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
@torch.no_grad()
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
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
        model.epoch_end(epoch, result)
        history.append(result)
    return history
def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
class Model(ImageClassificationBase):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, output_size)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        out = F.relu(out)
        out = self.linear6(out)
        return out
model = Model(784, 2048, 26)
history = [evaluate(model, val_loader)]
history
sessions = [
    [15, 1e-3],
    [20, 1e-4],
    [20, 1e-5],
    [20, 1e-6],
]
ctr = 1
for epochs, lr in sessions:
    print("="*80, end='\n\n')
    print(f"Currently in session {ctr}")
    print(f"Current learning rate: {lr}")
    print(f"Number of epochs to run for: {epochs}")
    history += fit(epochs, lr, model, train_loader, val_loader)
    ctr += 1
    print('\n')
result = evaluate(model, test_loader)
result
torch.save(model.state_dict(), 'MNIST-SIGN-LANGUAGE.pth')
lrs = [x[0] for x in sessions]
epochs = [x[1] for x in sessions]
arch = f"6 layers (784, 2048, 256, 128, 64, 32, 26)"
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project="MNIST-SIGN-LANGUAGE", environment=None)
jovian.reset()
jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)
jovian.log_metrics(test_loss=result['val_loss'], test_acc=result['val_acc'])
jovian.commit(project="MNIST-SIGN-LANGUAGE", outputs=['MNIST-SIGN-LANGUAGE.pth'], environment=None)
