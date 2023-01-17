# Uncomment and run the commands below if imports fail
!conda install numpy pytorch torchvision cpuonly -c pytorch -y
!pip install matplotlib --upgrade --quiet
!pip install pandas --upgrade --quiet
!pip install seaborn --upgrade --quiet
!pip install jovian --upgrade --quiet
import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
%matplotlib inline
project_name='forest-fires-regression-prediction' # will be used by jovian.commit
df_raw = pd.read_csv('../input/forest-fires-data-set-portugal/forestfires.csv')
df_raw.head()
df = df_raw.drop(['X','Y','month','day','area','DMC','DC'], axis=1)
df.head()
# Compute the correlation matrix
corr_matrix = df.drop(['ISI'], axis=1).corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(200, 10, as_cmap=True)
sns.heatmap(corr_matrix, cmap=cmap, annot=True,linewidth = 0.1,  cbar_kws={"shrink": .5})
corr_with_isi = df.corr()['ISI'].sort_values(ascending=False)
plt.figure(figsize=(5,4))
corr_with_isi.drop('ISI').plot.bar()
plt.show();
sns.pairplot(df[['FFMC', 'temp', 'RH', 'wind', 'rain']])
plt.show()
num_rows = df.shape[0]
num_rows
print(df.columns)
#cols number
print(df.shape[1])
input_cols = ['FFMC','temp','RH','wind','rain']
output_cols = ['ISI']
jovian.commit(project=project_name, environment=None)
#using pytorch as framework, so we have to convert all to pytorch format
def dataframe_to_arrays(dataframe):
    #copying dataframe for later use
    dataframe1 = dataframe.copy(deep=True)
    #converting to numpy
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array
inputs_array.shape , targets_array.shape
inputs_array, targets_array = dataframe_to_arrays(df)
inputs_array.shape , targets_array.shape
#converting numpy arrays to pytorch format
inputs = torch.from_numpy(inputs_array).float()
targets = torch.from_numpy(targets_array).float()
inputs.shape, targets.shape
dataset = TensorDataset(inputs, targets)
val_percent = 0.19
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size

train_ds, val_ds = random_split(dataset, [train_size,val_size])
batch_size = 32
#creating training and validation loader
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
input_size = len(input_cols)
output_size = len(output_cols)
input_size , output_size
class FFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, xb):
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out, targets)
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out, targets)
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 10th epoch
        if (epoch+1) % 10 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
ffmodel = FFModel()
list(ffmodel.parameters())
jovian.commit(project=project_name, environment=None)
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
            #zero-grad for setting all parameters to zero , for new training phrase
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history
result = evaluate(ffmodel, val_loader)
print(result)
epochs = 1000
lr = 3e-4
opt_func = torch.optim.Adam
history1 = fit(epochs, lr, ffmodel, train_loader, val_loader , opt_func)
def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)
from random import sample

# showing all predicted , target given and input 
for i in sample(range(0,len(val_ds)),10):
    input, target = val_ds[i]
    predict_single(input, target, ffmodel)
    print()
jovian.commit(project=project_name, environment=None)
