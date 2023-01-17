# Uncomment and run the commands below if imports fail
# !conda install numpy pytorch torchvision cpuonly -c pytorch -y
# !pip install matplotlib --upgrade --quiet
!pip install jovian --upgrade --quiet
import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
project_name='02-insurance-linear-regression' # will be used by jovian.commit
DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
DATA_FILENAME = "insurance.csv"
download_url(DATASET_URL, '.')
dataframe_raw = pd.read_csv(DATA_FILENAME)
dataframe_raw.head()
your_name = "sanath" # at least 5 characters
def customize_dataset(dataframe_raw, rand_str):
    dataframe = dataframe_raw.copy(deep=True)
    # drop some rows
    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))
    # scale input
    dataframe.bmi = dataframe.bmi * ord(rand_str[1])/100.
    # scale target
    dataframe.charges = dataframe.charges * ord(rand_str[2])/100.
    # drop column
    if ord(rand_str[3]) % 2 == 1:
        dataframe = dataframe.drop(['region'], axis=1)
    return dataframe
dataframe = customize_dataset(dataframe_raw, your_name)
dataframe.head()
num_rows = dataframe.shape[0]
print(num_rows)
num_cols = dataframe.shape[1]
print(num_cols)
input_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'charges']
categorical_cols = ['sex', 'smoker']
output_cols = ['charges']
# Write your answer here
jovian.commit(project=project_name, environment=None)
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
inputs_array, targets_array
inputs = inputs_array
targets = targets_array
inputs.dtype, targets.dtype
tensor_x = torch.Tensor(inputs) # transform to torch tensor
tensor_y = torch.Tensor(targets)

dataset = TensorDataset(tensor_x,tensor_y)
val_percent = 0.1 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
# train_ds, val_ds = ??? # Use the random_split function to split dataset into 2 parts of the desired length
batch_size = 32
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break
jovian.commit(project=project_name, environment=None)
input_size = len(input_cols)
output_size = len(output_cols)
class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 =  nn.Linear(input_size,40) 
        self.linear2 =  nn.Linear(40,output_size) 
        
        # fill this (hint: use input_size & output_size defined above)
        
    def forward(self, xb):
        x1 = F.relu(self.linear1(xb))
        x2 = F.relu(self.linear2(x1))
        out = x2                      # fill this
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.mse_loss(out, targets)                          # fill this
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.mse_loss(out, targets)                           # fill this    
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
model = InsuranceModel()
list(model.parameters())
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
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history
result = evaluate(model, val_loader) # Use the the evaluate function
print(result)
epochs = 10
lr = 0.01
history1 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 20
lr = .001
history2 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 30
lr = .00001
history3 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 40
lr = .000001
history4 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 100
lr = .00001
history5 = fit(epochs, lr, model, train_loader, val_loader)
val_loss = 428244992.0000
jovian.log_metrics(val_loss=val_loss)
jovian.commit(project=project_name, environment=None)
def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model.forward(inputs)                # fill this
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)
input, target = val_ds[0]
predict_single(input, target, model)
input, target = val_ds[10]
predict_single(input, target, model)
input, target = val_ds[23]
predict_single(input, target, model)
jovian.commit(project=project_name, environment=None)
jovian.commit(project=project_name, environment=None) # try again, kaggle fails sometimes
