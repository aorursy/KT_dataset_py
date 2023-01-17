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
path = "/kaggle/input/lhcb-jet-data-separation/"

signalData = pd.read_csv(path+"bjet_train.csv") # signal has mc_flavour = 5

backgroundData = pd.concat([pd.read_csv(path+"cjet_train.csv"), 

                            pd.read_csv(path+"ljet_train.csv")]) # background has mc_flavour != 5

print("First of {} signal rows".format(signalData.shape[0]))

display(signalData.iloc[0])

print("First of {} background rows".format(backgroundData.shape[0]))

display(backgroundData.iloc[0])
%matplotlib inline

import matplotlib.pyplot as plt

plotCols = list(signalData.columns)



for i in range(len(plotCols)):

    print("Plotting {}".format(plotCols[i]))

    plt.hist(signalData[plotCols[i]],label = "Sig")

    plt.hist(backgroundData[plotCols[i]],label = "Bkg")

    plt.legend()

    plt.xlabel(plotCols[i])

    plt.show()



# Note if the plots do not display minimise then maximise the output area below (double arrow button to the top right)
# Try fdChi2 as log10, others as linear

logCol = ['fdChi2']

linCol = ['PT', 'ETA', 'drSvrJet', 'fdrMin', 

          'm', 'mCor', 'mCorErr', 'pt', 'ptSvrJet',

          'tau', 'ipChi2Sum', 'nTrk', 'nTrkJet'] # Note skip Id as that is not helpful

# redefine columns as log10(col), so ranges are more similar between variables

for l in logCol:

    signalData[l] = np.log10(signalData[l])

    backgroundData[l] = np.log10(backgroundData[l])



nTrainSig = signalData.shape[0]//2 # half the rows for training, half for evaluation

nTrainBkg = backgroundData.shape[0]//2



# first half as training

x_data = np.concatenate([signalData[logCol+linCol][:nTrainSig].values,

                         backgroundData[logCol+linCol][:nTrainBkg].values])

y_data = np.concatenate([(signalData["mc_flavour"][:nTrainSig]==5).values.astype(np.int),

                         (backgroundData["mc_flavour"][:nTrainBkg]==5).values.astype(np.int)])



#second half as evaulation

x_eval = np.concatenate([signalData[logCol+linCol][nTrainSig:].values,

                         backgroundData[logCol+linCol][nTrainBkg:].values])

y_eval = np.concatenate([(signalData["mc_flavour"][nTrainSig:]==5).values.astype(np.int),

                         (backgroundData["mc_flavour"][nTrainBkg:]==5).values.astype(np.int)])
import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import datasets, transforms

from torch.autograd import Variable

import torch.utils.data as Data



import pylab as pl

import time
epochs        = 100                      # number of training epochs

batch_size    = 32                       # number of samples per batch

input_size    = len(logCol+linCol)                        # The number of features

num_classes   = 2                        # The number of output classes. In this case: [star, galaxy, quasar]

hidden_size   = 5                        # The number of nodes at the hidden layer

learning_rate = 1e-3                     # The speed of convergence

verbose       = True                     # flag for printing out stats at each epoch
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler() # initialise StandardScaler



scaler.fit(x_data) # Fit only to the training data



x_data_scaled = scaler.transform(x_data)

x_eval_scaled = scaler.transform(x_eval)
x_data_scaled  = torch.tensor(x_data_scaled, dtype=torch.float)

y_data  = torch.tensor(y_data, dtype=torch.long)

x_eval_scaled  = torch.tensor(x_eval_scaled, dtype=torch.float)

y_eval  = torch.tensor(y_eval, dtype=torch.long)



x_data_scaled, y_data = Variable(x_data_scaled), Variable(y_data)

x_eval_scaled, y_eval = Variable(x_eval_scaled), Variable(y_eval)



train_data = Data.TensorDataset(x_data_scaled, y_data)

valid_data = Data.TensorDataset(x_eval_scaled, y_eval)



train_loader = Data.DataLoader(dataset=train_data,

                               batch_size=batch_size,

                               shuffle=True)



valid_loader = Data.DataLoader(dataset=valid_data,

                               batch_size=batch_size,

                               shuffle=True)
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
start = time.time()



_results = []

for epoch in range(epochs):  # loop over the dataset multiple times



    # training loop for this epoch

    model.train() # set the model into training mode

    

    train_loss = 0.

    for batch, (x_data_scaled, y_data) in enumerate(train_loader):

        

        model.zero_grad()

        out, prob = model(x_data_scaled)

        

        loss = F.cross_entropy(out, y_data)

        

        loss.backward()

        optimizer.step()

        

        train_loss += loss.item() * x_data_scaled.size(0)

    

    train_loss/= len(train_loader.dataset)



    if verbose:

        print('Epoch: {}, Train Loss: {:4f}'.format(epoch, train_loss))



    # validation loop for this epoch:

    model.eval() # set the model into evaluation mode

    with torch.no_grad():  # turn off the gradient calculations

        

        correct = 0; valid_loss = 0

        for i, (x_eval_scaled, y_eval) in enumerate(valid_loader):

            

            out, prob = model(x_eval_scaled)

            loss = F.cross_entropy(out, y_eval)

            

            valid_loss += loss.item() * x_eval_scaled.size(0)

            

            preds = prob.argmax(dim=1, keepdim=True)

            correct += preds.eq(y_eval.view_as(preds)).sum().item()

            

        valid_loss /= len(valid_loader.dataset)

        accuracy = correct / len(valid_loader.dataset)



    if verbose:

        print('Validation Loss: {:4f}, Validation Accuracy: {:4f}'.format(valid_loss, accuracy))



    # create output row:

    _results.append([epoch, train_loss, valid_loss, accuracy])



results = np.array(_results)

print('Finished Training')

print("Final validation error: ",100.*(1 - accuracy),"%")



end = time.time()

print("Run time [s]: ",end-start)
print(results.shape)



pl.subplot(111)

pl.plot(results[:,0],results[:,1], label="training")

pl.plot(results[:,0],results[:,2], label="validation")

pl.xlabel("Epoch")

pl.ylabel("Loss")

pl.legend()

pl.show()
testData = pd.read_csv(path+"competitionData.csv")

# apply log10 to columns that need it

for l in logCol:

    testData[l] = np.log10(testData[l])

x_test = testData[logCol+linCol].values



x_test_scaled = scaler.transform(x_test)



predMCFloat = model.predict(x_test_scaled)

# predMCFloat is a float: need to convert to an int

predMC = (predMCFloat>0.5).astype(np.int)

testData["Prediction1"] = predMC



# solution to submit

display(testData[["Id","Prediction1"]]) # display 5 rows

# write to a csv file for submission

testData.to_csv("submit.csv.gz",index=False,columns=["Id","Prediction1"],compression="gzip") # Output a compressed csv file for submission: see /kaggle/working to the right