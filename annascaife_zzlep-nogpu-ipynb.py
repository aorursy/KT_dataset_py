import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import pandas as pd
import pylab as pl
import time
samples       = ['ggH125_ZZ4lep','llll'] # datafiles for input
epochs        = 10                       # number of training epochs
batch_size    = 32                       # number of samples per batch
input_size    = 2                        # The number of features
num_classes   = 2                        # The number of output classes
hidden_size   = 10                       # The number of perceptrons in the hidden layer
learning_rate = 1e-3                     # The speed of convergence
verbose       = True                     # flag for printing out stats at each epoch
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ", device)
DataFrames = {} # define empty dictionary to hold dataframes
for s in samples: # loop over samples
    DataFrames[s] = pd.read_csv('/kaggle/input/4lepton/'+s+".csv") # read .csv file
# cut on lepton charge
def cut_lep_charge(lep_charge_0,lep_charge_1,lep_charge_2,lep_charge_3):
# only want to keep events where sum of lepton charges is 0
    sum_lep_charge = lep_charge_0 + lep_charge_1 + lep_charge_2 + lep_charge_3
    if sum_lep_charge==0: return True
    else: return False

# apply cut on lepton charge
for s in samples:
    # cut on lepton charge using the function cut_lep_charge defined above
    DataFrames[s] = DataFrames[s][ np.vectorize(cut_lep_charge)(DataFrames[s].lep_charge_0,
                                                    	    DataFrames[s].lep_charge_1,
                                                    	    DataFrames[s].lep_charge_2,
                                                    	    DataFrames[s].lep_charge_3) ]
ML_inputs = ['lep_pt_1','lep_pt_2'] # list of features for ML model
all_MC = [] # define empty list that will contain all features for the MC

for s in samples: # loop over the different samples
    if s!='data': # only MC should pass this
        all_MC.append(DataFrames[s][ML_inputs]) # append the MC dataframe to the list containing all MC features
        
X = np.concatenate(all_MC) # concatenate the list of MC dataframes into a single 2D array of features, called X
all_y = [] # define empty list that will contain labels whether an event in signal or background

for s in samples: # loop over the different samples
    if s!='data': # only MC should pass this
        if 'H125' in s: # only signal MC should pass this
            all_y.append(np.ones(DataFrames[s].shape[0])) # signal events are labelled with 1
        else: # only background MC should pass this
            all_y.append(np.zeros(DataFrames[s].shape[0])) # background events are labelled 0
            
y = np.concatenate(all_y) # concatenate the list of lables into a single 1D array of labels, called y
from sklearn.model_selection import train_test_split

# make train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                  test_size=0.33, 
                                                  random_state=492 ) # set the random seed for reproducibility
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # initialise StandardScaler

scaler.fit(X_train) # Fit only to the training data

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)
X_train  = torch.tensor(X_train, dtype=torch.float)
y_train  = torch.tensor(y_train, dtype=torch.long)

X_train, y_train = Variable(X_train), Variable(y_train)

x_valid, y_valid = X_train[:100], y_train[:100]
x_train_nn, y_train_nn = X_train[100:], y_train[100:]

train_data = Data.TensorDataset(x_train_nn, y_train_nn)
valid_data = Data.TensorDataset(x_valid, y_valid)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=batch_size,
                               shuffle=True,
                               **kwargs)

valid_loader = Data.DataLoader(dataset=valid_data,
                               batch_size=batch_size,
                               shuffle=True,
                               **kwargs)
class Classifier_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        
        self.h1  = nn.Linear(in_dim, hidden_dim)
        self.h2  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.out(x)
        
        return x, F.softmax(x, dim=1)
model = Classifier_MLP(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model = model.to(device)

start = time.time()

_results = []
for epoch in range(epochs):  # loop over the dataset multiple times

    # training loop for this epoch
    model.train() # set the model into training mode
    
    train_loss = 0.
    for batch, (x_train, y_train) in enumerate(train_loader):
        
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        model.zero_grad()
        out, prob = model(x_train)
        
        loss = F.cross_entropy(out, y_train)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * x_train.size(0)
    
    train_loss/= len(train_loader.dataset)

    if verbose:
        print('Epoch: {}, Train Loss: {:4f}'.format(epoch, train_loss))

    # validation loop for this epoch:
    model.eval() # set the model into evaluation mode
    with torch.no_grad():  # turn off the gradient calculations
        
        correct = 0.; valid_loss = 0.
        for i, (x_valid, y_valid) in enumerate(valid_loader):
            
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            
            out, prob = model(x_valid)
            loss = F.cross_entropy(out, y_valid)
            
            valid_loss += loss.item() * x_valid.size(0)
            
            preds = prob.argmax(dim=1, keepdim=True)
            correct += preds.eq(y_valid.view_as(preds)).sum().item()
            
        valid_loss /= len(valid_loader.dataset)
        accuracy = correct / len(valid_loader.dataset)

    if verbose:
        print('Validation Loss: {:4f}, Validation Accuracy: {:4f}'.format(valid_loss, accuracy))

    # create output row:
    _results.append([epoch, train_loss, valid_loss, accuracy])

results = np.array(_results)
print('Finished Training')
print("Final validation error: ",100.*(1 - accuracy),"%")

if use_cuda: torch.cuda.synchronize() 
end = time.time()
print("Run time [s]: ",end-start)
pl.subplot(111)
pl.plot(results[:,0],results[:,1], label="training")
pl.plot(results[:,0],results[:,2], label="validation")
pl.xlabel("Epoch")
pl.ylabel("Loss")
pl.legend()
pl.show()