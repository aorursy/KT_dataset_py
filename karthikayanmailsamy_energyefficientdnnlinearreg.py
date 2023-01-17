#It is a good practice to import all the modules required for training the model.
import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx" #The url of the dataset is given here
DATA_FILENAME = "Energyefficiency.csv" #the name of the saved downloaded file.
download_url(DATASET_URL, root='.',filename=DATA_FILENAME) #Don't forget to add the directory(any name where the downloaded file is saved) at the root parameter.
# To read the data from the excel file(.xlsx means excel),we use pandas dataframe where dataframe is like row by column which can be visualized below.
dataframe = pd.read_excel(DATA_FILENAME) 
dataframe.head(5) #It is used to get the first 5 rows of the dataframe.
num_rows = len(dataframe)
print(num_rows)
num_cols = sum(1 for i in dataframe.columns)
print(num_cols)
input_cols = [i for i in dataframe.columns if(i!='Y1' and i!='Y2')] #the column titles of the input variables
print(len(input_cols))
output_cols = ['Y1','Y2'] #the column titles of output/target variable(s)
print(len(output_cols))
# Visualizing the distribution of target values in a graph.
plt.plot(dataframe['Y1'],'r')
plt.plot(dataframe['Y2'],'b')
plt.legend(['heating load','cooling load'])
plt.show()
import numpy as np
def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe as we may need the original one if anything goes wrong which is a good practice.
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays of datatype float32 as the model expects the data to be float instead of double.
    inputs_array = dataframe1[input_cols].to_numpy().astype(np.float32)
    targets_array = dataframe1[output_cols].to_numpy().astype(np.float32)
    return inputs_array, targets_array
inputs_array, targets_array = dataframe_to_arrays(dataframe)
inputs_array, targets_array
# We should convert the numpy arrays into tensor as the model expects tensors.
inputs = torch.from_numpy(inputs_array)
targets = torch.from_numpy(targets_array)
#let's confirm that the datatype is float
inputs.dtype, targets.dtype
# Let's now create the dataset with inputs and targets combined to create batches at next step
dataset = TensorDataset(inputs, targets)
# It is necessary to split the data into train and validation in order to evaluate the model on untouched data while training the model.
torch.manual_seed(16)#this will make the random same everytime we run this notebook which will help us evaluate a particular value as we do below .
val_percent = 0.1 # between 0.1 and 0.2 is good.
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_ds, val_ds = random_split(dataset,[train_size,val_size]) # Use the random_split function to split dataset into 2 parts of the desired length.
#Usually we take batch_size as a power of 2(i.e,16,32,64,...) as it improves the speed of the traing convergence(i.e,loss is reduced to minimum soon)
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True) #pin_memory enables the data to be loaded into pinmemory which speedups the GPU loading action.
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True) # num_works are used when the dataset is large to make the loading fat as it is done parallely by 4 systems.
#Let's have a look at the first batch.
for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break
input_size = len(input_cols)
output_size = len(output_cols)
class EfficiencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #input layer
        self.linear1=nn.Linear(input_size,128)  # we have to pass the number of inputs(number of input columns) and the number of hidden units
        #hidden layers
        self.linear2=nn.Linear(128,64) #previous layer hidden units count and the current layers hidden units count
        self.linear3=nn.Linear(64,32) # make sure that the previous layer hidden units count and the current layer hidden units count are equal.
        self.linear4=nn.Linear(32,16)
        self.linear6=nn.Linear(16,8)
        #output layer
        self.linear5=nn.Linear(8,output_size) # here we need to pass previous layers hidden units count and number of outputs(number of output columns) as we mentioned above.
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1) #As the model expects the features to flattened(i.e, all the features are stacked vertically),view()will do this for us where -1 makes the model to rest features as 2nd dimension.
        # Apply layers & activation functions
        out= self.linear1(out) #It is an inbuilt-method for linear regression i.e  z= w*x +b where w is the weight and b is the bias
        out=F.relu(out)  #As we can know from the previous post that linear relation can't make a good prediction , now we are using a non-linear activation function called rectified linear units (Relu)
        out= self.linear2(out)
        out=F.relu(out)
        out= self.linear3(out)
        out=F.relu(out)
        out= self.linear4(out)
        out=F.relu(out)
        out= self.linear6(out)
        out=F.relu(out)
        out=self.linear5(out)  # here we used 5 layers including 3 hidden layers.
        return out
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)  #which returns the predicted values      
        # Calcuate loss
        loss = F.l1_loss(out,targets) #L1 loss(i.e ((1/num_rows)*(predicted-target)^2)^1/2) which is used to calculate the loss. Although there are lot loss functions, l1_loss will be more suitable for linear reg.
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out,targets)                           # l1_loss is calculated for validation set   
        return {'val_loss': loss.detach()}  
        #detach() is used to remove this from computational graph as it may confuse the backprop since it is the loss of validation set to clear the memory whic will make training fast.
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combining losses for all the batches to compute a single loss value
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch where epoch represents a complete traversal through the training examples.
        if (epoch+1) % 2 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch+1, result['loss'],result['val_loss']))
torch.cuda.is_available() #This gives whether the gpu is available or not in form of boolean(true or false)
def get_default_device():
    """To pick GPU if available, else CPU so that the program won't crash if gpu not available"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device=get_default_device()
print(device)
def to_device(data, device):
    """Move tensor(s) to chosen device as everything should be present in the respective device """
    if isinstance(data, (list,tuple)): # isinstance checks whether the input is of the desired type like it is list or tuple and will go next only if the condition is passed.
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True) #thus the data is moved according to the device selected

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device .Don't worry if you dont't get it,just do as instructed"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
#The above class will perform the functions when the instance is called.
#Let's now create the instances for the below data.
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
model = to_device(EfficiencyModel(),device) #Initializing the model and also moving the model to the device
#once the model is initialized ,all the w's(weights) and b's(biases) are initialized at random which are then updated by calculating the derivative of the loss fuction with respect to both w and b respectively.
#This is how the model is made to learn the relations between inputs and targets.
list(model.parameters())
def evaluate(model, val_loader):
    #To evaluate the validation set
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr) #optimizer which is used update the parameters. Stochastic Gradient Descent is mostly used for linear reg. problems.
    for epoch in range(epochs):
        # Training Phase 
        losses=[]
        for batch in train_loader:
            loss = model.training_step(batch)
            losses.append(loss) #the training losses are stored for every batch for future visualizations.
            loss.backward() # Backpropagation(i.e, derivative(slope) calculation) is made by the model.
            optimizer.step() #It updates the weights and biases with the derivatives and learning rate(i.e, w=w-lr*dw)
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['loss']= torch.stack(losses).mean()  #The mean of training losses is stored for each epoch.
        model.epoch_end(epoch, result, epochs)
        history.append(result) #appending the results for visualization purposes.
    return history
result = evaluate(model,val_loader) # Use the the evaluate function to evaluate the model.
print(result)
epochs = 150 #number of epochs to run the model.
lr = 0.5e-3  #learning rate
model = to_device(EfficiencyModel(),device)
model.load_state_dict(torch.load("savedmodel/1")) #To load the model and predict
model.eval()
#If you want run comment the above 2 lines and uncomment the below line
#history = fit(epochs, lr, model, train_loader, val_loader)
val_loss = evaluate(model,val_loader)
print(val_loss)
#Let's visualize the validation loss for the epochs
plt.plot([i['loss'] for i in history],'r')
plt.plot([i['val_loss'] for i in history],'b')
plt.xlabel("epochs")
plt.ylabel("losses")
plt.legend(['loss','val_loss'])
plt.show()
def predict_single(input, target, model):
    predictions = model(input.unsqueeze(0))     
    prediction = predictions[0].detach() #detach() is done to remove it from the computational graphs i.e, its dervatives are no longer calculated.
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)
#Let's check the predictions of the model by passing a value from the validation set(you can also try by passing a tensor of your wish as torch.tensor([...]))
print(val_ds[4])
val=DeviceDataLoader(val_ds[4],device) #we have to move the data to the device to avoid errors.
input, target = val
predict_single(input, target, model)