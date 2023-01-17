import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
dataframe = pd.read_csv('../input/bank-marketing-campaigns-dataset/bank-additional-full.csv',sep=';')
dataframe.head()
num_rows = dataframe.shape[0]
print('Number of instances:', num_rows)
num_cols = dataframe.shape[1]
print('Number of columns:', num_cols)
input_cols = dataframe.columns.difference(['y'])
print('Input variables:', input_cols)
output_cols = ['y']
print('Output variable:', output_cols)
categorical_cols = dataframe.columns.difference(dataframe._get_numeric_data().columns)
print('Categorical variables:', categorical_cols)

import seaborn as sns
# Configuring styles
sns.set_style("darkgrid")
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['figure.facecolor'] = '#00000000'

plt.title("Distribution of Subscriptions")
sns.countplot(x="y", data=dataframe, palette="bwr")

def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array
inputs_array, targets_array = dataframe_to_arrays(dataframe)
inputs = torch.tensor(inputs_array, dtype= torch.float32)
targets = torch.tensor(targets_array, dtype= torch.long)
targets = targets.squeeze(dim=1)
inputs,targets
dataset = TensorDataset(inputs, targets)    

val_percent = 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size

train_ds, val_ds = random_split(dataset, (train_size, val_size))
batch_size = 32
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size,)

class InsuranceModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, xb):
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        out = self(inputs)                       
        loss = F.cross_entropy(out, targets)
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = F.cross_entropy(out,targets)
        acc = accuracy(out, targets)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch+1, result['val_loss'], result['val_acc']))
input_size = len(input_cols)
output_size = 2
model = InsuranceModel(input_size, output_size)
model
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history
            
result = evaluate(model, val_loader) # Use the the evaluate function
print(result)
epochs = 10
lr = 0.01
history1 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 50
lr = 0.001
history1 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 100
lr = 0.01
history1 = fit(epochs, lr, model, train_loader, val_loader)
val_loss = evaluate(model, val_loader)
val_loss
def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)
input, target = val_ds[1]
predict_single(input, target, model)