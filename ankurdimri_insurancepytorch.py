import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
DATA_FILENAME = "insurance.csv"
download_url(DATASET_URL, '.')
dataframe = pd.read_csv(DATA_FILENAME)
dataframe.head()
num_rows = dataframe.shape[0]
print(num_rows)
num_cols = dataframe.shape[1]
print(num_cols)
dataframe.columns
input_cols = ['age', 'sex', 'bmi', 'children', 'smoker','region']
categorical_cols = ['sex', 'smoker','region']
output_cols = ['charges']
output = dataframe[output_cols];
output.head()
print(f'Maximum Value: {round(output.max(),2)} / Minimum Value: {round(output.min(),2)} / Average Value: {round(output.mean(),2)}')
# Import libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.title("Distribution of Charges")

sns.distplot(output,bins=10);
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
inputs = torch.tensor(inputs_array,dtype=torch.float32)
targets = torch.tensor(targets_array,dtype=torch.float32)
inputs.dtype, targets.dtype
dataset = TensorDataset(inputs, targets)
val_percent = 0.15 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_ds, val_ds = random_split(dataset, [train_size, val_size]) # Use the random_split function to split dataset into 2 parts of the desired length
batch_size = 16
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break
input_size = len(input_cols);print(input_size)
hidden_size = 16;print(hidden_size)
output_size = len(output_cols);print(output_size)
class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Dropout(p=0.2)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden_size,output_size)
        
    def forward(self, xb):
        out = self.layer1(xb)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out,targets)                          
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out,targets)                                
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
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    model.train()
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
epochs = 200
lr = (1e0)
history1 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 200
lr = 1e-1
history2 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 150
lr = 1e-2
history3 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 100
lr = 1e-3
history4 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 50
lr = 1e-5
history5 = fit(epochs, lr, model, train_loader, val_loader)
# Replace these values with your results
loss = [r['val_loss'] for r in [result] + history1 + history2 + history3 + history4 + history5]
plt.plot(loss, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs');
val_loss = loss[-1]
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
input, target = val_ds[44]
predict_single(input, target, model)
