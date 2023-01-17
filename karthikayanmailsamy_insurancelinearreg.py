#It is a good practice to import all the modules that may be needed for the project.
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
dataframe_raw= pd.read_csv(DATA_FILENAME) #To read csv from a csv file using pandas dataframe where dataframe is like rows by columns.
dataframe_raw.head(5) #used to see first 5 data in the dataframe
your_name = "karthikayan" # at least 5 characters
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
num_cols = sum(1 for i in dataframe.columns)
print(num_cols)
input_cols = [i for i in dataframe.columns if(i!='charges')] #to get the input column titles
print(input_cols)
#to get the columns that contain string values instead of float values(will explain why we are doing this later)
categorical_cols = [i for i in input_cols if(type(dataframe[i][0]) is str)] 
print(categorical_cols)
output_cols = ['charges'] #the output column titles are obtained here.
mini=min(dataframe['charges'])
maxi=max(dataframe['charges'])
mean=sum(dataframe['charges'])/(len(dataframe['charges']))
print(mini,maxi,mean)
plt.plot([i for i in range(num_rows)],dataframe['charges'],'x-r')
plt.title("Output Distribution")
plt.xlabel('people')
plt.ylabel('charges')
plt.show()
import numpy as np
def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe in order to make changes to the dataframe according to our needs
    dataframe1 = dataframe.copy(deep=True) 
    # Convert non-numeric categorical columns to numbers (i.e, string variables to numbers as we can communicate with the machine only with numbers)
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes 
        #used to convert string to numbers(i.e if a column contains 'yes' or 'no' predominantly,it will assign 1-'yes' and 0-'no' likewise)
        #print(dataframe1[col][:10],dataframe[col][:10])
    # Extract input & outupts as numpy arrays in datatype float32 as the tensor expects the data to be of float type.
    inputs_array = dataframe1[input_cols].to_numpy().astype(np.float32)
    targets_array = dataframe1[output_cols].to_numpy().astype(np.float32)
    return inputs_array, targets_array
inputs_array, targets_array = dataframe_to_arrays(dataframe)#you can understand better on looking into the outputs
inputs_array, targets_array 
#now the numpy arrays are converted to tensors
inputs = torch.from_numpy(inputs_array)
targets = torch.from_numpy(targets_array)
inputs.dtype, targets.dtype
dataset = TensorDataset(inputs, targets) #now the input and outputs are combined
dataset[0] #you can see that the first part contains input feature values and the second part contains output values
#let's now split the train data into train and validation so that we can see the performance on the untouched data parallely
val_percent = 0.1 # may be between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_ds, val_ds = random_split(dataset,[train_size,val_size]) # Use the random_split function to split dataset into 2 parts of the desired length
batch_size = 128 # it is used to speed up the learning process
#we are shuffling the data in order to avoid overfitting as the input data is contiguous in aspect of categories(i.e, initial half of data represents a category and as it goes on other categories)
train_loader = DataLoader(train_ds, batch_size, shuffle=True) 
val_loader = DataLoader(val_ds, batch_size)
#let's have a look at the 1st batch 
for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break
input_size = len(input_cols)
output_size = len(output_cols)
print(input_size,output_size)
class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)                  # fill this (hint: use input_size & output_size defined above)
        
    def forward(self, xb):
        out = self.linear(xb)                         # fill this
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out,targets)                          # fill this
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out,targets)                           # fill this    
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
result = evaluate(model,val_loader) # Use the the evaluate function
print(result)
epochs = 100
lr = 1e-4
history1 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 100
lr = 1e-5
model = InsuranceModel()
history2 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 100
lr = 1e-6
model = InsuranceModel()
history3 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 100
lr = 1e-7
model = InsuranceModel()
history4 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 40000
lr = 1e-4
model = InsuranceModel()
history5 = fit(epochs, lr, model, train_loader, val_loader)
val_loss = history5[-1]['val_loss']
print(val_loss)
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
input, target = val_ds[20]
predict_single(input, target, model)
