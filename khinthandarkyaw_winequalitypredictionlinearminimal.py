# Uncomment and run the commands below if imports fail
!conda install numpy pytorch torchvision cpuonly -c pytorch -y
!pip install matplotlib --upgrade --quiet
!pip install jovian --upgrade --quiet
# Imports
import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
# Hyperparameters
batch_size=64
learning_rate=5e-7


# Other constants
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_FILENAME = "winequality-red.csv"
TARGET_COLUMN = 'quality'
input_size=11
output_size=1
# Download the data
download_url(DATASET_URL, '.')
dataframe = pd.read_csv(DATA_FILENAME, delimiter = ';') # here you have to use delimiter ';' which will make your raw data into tabular form with colums and rows
dataframe.head()
dataframe.tail()
# Convert from Pandas dataframe to numpy arrays
inputs = dataframe.drop('quality', axis=1).values # axis = 1 means column
targets = dataframe[['quality']].values
inputs.shape, targets.shape
# Convert to PyTorch dataset
dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
train_ds, val_ds = random_split(dataset, [1000, 599]) # we have total 1599

train_loader = DataLoader(train_ds, batch_size, shuffle=True) # we have to shuffle train_ds to get better accuracy
val_loader = DataLoader(val_ds, batch_size*2) # here we use batch_size*2
class WineModel(nn.Module):
    def __init__(self):
        super().__init__() # base class object initialization
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, xb): # initial loading for input data
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        out = self(inputs)                 # Generate predictions
        loss = F.mse_loss(out, targets)    # Calculate loss
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch 
        out = self(inputs)                 # Generate predictions
        loss = F.mse_loss(out, targets)    # Calculate loss
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result): # to show epoch
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))
    
model = WineModel()
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
result = evaluate(model, val_loader) # result before training
result
history = fit(1000, learning_rate, model, train_loader, val_loader) # training with 1000 epochs
losses = [r['val_loss'] for r in [result] + history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('val_loss')
plt.title('val_loss vs. epochs');
def predict_single(x, model):
    xb = x.unsqueeze(0)
    return model(x).item()
x, target = val_ds[10]
pred = predict_single(x, model)
print("Input: ", x)
print("Target: ", target.item())
print("Prediction:", pred)
torch.save(model.state_dict(), 'wine-linear.pth')
jovian.commit(project='wine-linear-minimal', environment=None, outputs=['wine-linear.pth'])
jovian.commit(project='wine-linear-minimal', environment=None, outputs=['wine-linear.pth']) # Kaggle commit fails sometimes, so try again..
