# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()
df.shape
import matplotlib.pyplot as plt
def plot_figure(index,column):
    plt.subplot(6,2,index)
    plt.title(column)
    plt.plot(df[column])
    
plt.figure(figsize=(10,10))

for index , column in enumerate(df.columns):
    if index+1<=len(df.columns):
        plot_figure(index+1, column)

plt.tight_layout()
df.dtypes
df.isnull().any()
import seaborn as sns
#correlation matrix
corrmat = df.corr()
k = 12 #number of variables for heatmap
cols = corrmat.nlargest(k, 'quality')['quality'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1)
plt.figure(figsize=(8,8))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
input_cols = list(df.columns)[:-1]
input_cols
output_cols = ['quality']
def dataframe_to_arrays(df):
    # Make a copy of the original dataframe
    df1 = df.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = df1[input_cols].to_numpy()
    targets_array = df1[output_cols].to_numpy()
    return inputs_array, targets_array
inputs_array, targets_array = dataframe_to_arrays(df)
inputs_array, targets_array
inputs_array.shape,targets_array.shape
import torch
inputs = torch.Tensor(inputs_array)
targets = torch.Tensor(targets_array)
from torch.utils.data import DataLoader, TensorDataset, random_split
dataset = TensorDataset(inputs, targets)
df.shape
num_rows = len(df)
val_percent = 0.01 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_df, val_df = random_split(dataset, [train_size, val_size]) 
batch_size = 50
train_loader = DataLoader(train_df, batch_size, shuffle=True)
val_loader = DataLoader(val_df, batch_size)
input_cols
output_cols
input_size = len(input_cols)
output_size = len(output_cols)
import torch.nn as nn
class WineModel(nn.Module):
    def __init__(self):
        super().__init__()     
        self.linear = nn.Linear(input_size, output_size) # fill this (hint: use input_size & output_size defined above)
        #model initialized with random weight
        
    def forward(self, xb):
        out = self.linear(xb)             # batch wise forwarding
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)         
        # Calcuate loss
        loss = F.l1_loss(out, targets)  # batch wise training step and loss
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss =F.l1_loss(out, targets)       # batch wise validation and loss    
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine val losses of all batches as average
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
model =  WineModel()
list(model.parameters())
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
        model.epoch_end(epoch, result, epochs)
        history.append(result)  #appends total validation loss of whole validation set epoch wise
    return history
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url

result = evaluate(model,val_loader) # Use the the evaluate function
print(result)
epochs = 1000
lr = 1e-2
history1 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 1000
lr = 1e-3
history2 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 1000
lr = 1e-4
history3 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 1000
lr = 1e-5
history3 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 1000
lr = 1e-6
history3 = fit(epochs, lr, model, train_loader, val_loader)
val_loss = evaluate(model,val_loader)
val_loss
def predict_single(input, target, model):
    inputs = input.unsqueeze(0) 
    predictions = model(inputs)
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)
input, target = val_df[0]
predict_single(input, target, model)
input, target = val_df[10]
predict_single(input, target, model)
input, target = val_df[5]
predict_single(input, target, model)