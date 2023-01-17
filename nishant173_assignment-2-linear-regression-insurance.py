# Uncomment and run the commands below if imports fail
# !conda install numpy pytorch torchvision cpuonly -c pytorch -y
# !pip install matplotlib --upgrade --quiet
!pip install jovian --upgrade --quiet
import jovian

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
from torchvision.datasets.utils import download_url

import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
project_name='02-insurance-linear-regression' # will be used by jovian.commit
DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
DATA_FILENAME = "insurance.csv"
download_url(DATASET_URL, '.')
dataframe_raw = pd.read_csv(DATA_FILENAME)
dataframe_raw.head()
your_name = 'Nishant' # at least 5 characters
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
num_rows = len(dataframe)
print(num_rows)
num_cols = dataframe.shape[1]
print(num_cols)
input_cols = dataframe.columns.drop(['charges']).tolist()
input_cols
categorical_cols = dataframe.select_dtypes(exclude=['int', 'float', 'bool']).columns.tolist()
categorical_cols
output_cols = ['charges']
output_cols
# Write your answer here
dataframe['charges'].describe()
plt.hist(dataframe['charges'], bins=20)
plt.title("Distribution of dependent variable (charges)")
plt.xlabel("charges (in unit currency)")
plt.ylabel("count")
plt.show()
jovian.commit(message='Assignment 2 - First commit', project=project_name, environment=None)
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
inputs = torch.tensor(inputs_array).type(torch.float32)
targets = torch.tensor(targets_array).type(torch.float32)
inputs.dtype, targets.dtype
dataset = TensorDataset(inputs, targets)
print(f" Inputs:\n{inputs} \n\n\n Targets:\n{targets}")
val_percent = 0.15 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_ds, val_ds = random_split(dataset=dataset, lengths=[train_size, val_size])
len(train_ds), len(val_ds)
batch_size = 128 # Can be increased later
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break
xb.shape
jovian.commit(message='Assignment 2 - Second commit', project=project_name, environment=None)
input_size = len(input_cols)
output_size = len(output_cols)
print(f"input_cols: {input_cols} \t output_cols: {output_cols}")
print(f"input_size: {input_size} \t output_size: {output_size}")
class InsuranceModel(nn.Module):
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
        # Print result every 20th epoch
        if (epoch+1) % 100 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
model = InsuranceModel()
list(model.parameters())
jovian.commit(message='Assignment 2 - Third commit', project=project_name, environment=None)
targets
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
model
train_loader, val_loader
%%time

learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
loss_by_learning_rate = dict()
for lr in learning_rates:
    history = fit(epochs=800, lr=lr, model=model, train_loader=train_loader, val_loader=val_loader)
    avg_loss = pd.np.mean([float(dict_loss['val_loss']) for dict_loss in history])
    loss_by_learning_rate[lr] = avg_loss
loss_by_learning_rate
val_loss = 7420.3687
jovian.log_metrics(val_loss=val_loss)
jovian.commit(message='Assignment 2 - Fourth and final commit', project=project_name, environment=None)
def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
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
