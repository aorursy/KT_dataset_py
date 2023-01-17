# This is a bit of code to make things work on Kaggle

import os

from pathlib import Path



if os.path.exists("/kaggle/input/ucfai-core-fa19-applications"):

    DATA_DIR = Path("/kaggle/input/ucfai-core-fa19-applications")

else:

    DATA_DIR = Path("./")
# import all the libraries you need



# torch for NNs

import torch 

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch import optim



# general imports

from sklearn.model_selection import train_test_split

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv(DATA_DIR / "master.csv")
dataset.head()
print("Total entries: {}, null entries: {}".format(len(dataset["HDI for year"]), dataset["HDI for year"].isnull().sum()))
dataset = dataset.drop("HDI for year", axis=1).drop("country-year", axis=1)

dataset.head()
dataset.describe()
dataset.info()
country_set = sorted(set(dataset["country"]))

country_map = {country : i for i, country in enumerate(country_set)}



sex_map = {'male': 0, 'female': 1}



age_set = sorted(set(dataset["age"]))

age_map = {age: i for i, age in enumerate(age_set)}



gen_set = sorted(set(dataset["generation"]))

gen_map = {gen: i for i, gen in enumerate(gen_set)}



def gdp_fix(x):

    x = int(x.replace(",", ""))

    return x



dataset = dataset.replace({"country": country_map, "sex": sex_map, "generation": gen_map, "age": age_map})

dataset[" gdp_for_year ($) "] = dataset.apply(lambda row: gdp_fix(row[" gdp_for_year ($) "]), axis=1)
dataset.head()
dataset.info()
dataset.describe()
dataset["year"] = (dataset["year"] - 1985) / 31

dataset["country"] = dataset["country"] / 100

dataset["age"] = dataset["age"] / 5

dataset["suicides_no"] = dataset["suicides_no"] / 22338

dataset["population"] = (dataset["population"] - 2.780000e2) / 4.380521e7

dataset[" gdp_for_year ($) "] = (dataset[" gdp_for_year ($) "] - 4.691962e7) / 1.812071e13

dataset["gdp_per_capita ($)"] = (dataset["gdp_per_capita ($)"] - 251) / 126352

dataset["generation"] = dataset["generation"] / 5


X, Y = dataset.drop("suicides/100k pop", axis=1).values, dataset["suicides/100k pop"].values

Y = np.expand_dims(Y, axis=1)

dataset.describe()
# Split data here using train_test_split

# YOUR CODE HERE

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)
print("X shape: {}, Y shape: {}".format(X.shape, Y.shape))

print(20e6)
# run this if you are using torch and a NN

class Torch_Dataset(Dataset):

  

  def __init__(self, data, outputs):

        self.data = data

        self.outputs = outputs



  def __len__(self):

        #'Returns the total number of samples in this dataset'

        return len(self.data)



  def __getitem__(self, index):

        #'Returns a row of data and its output'

      

        x = self.data[index]

        y = self.outputs[index]



        return x, y



# use the above class to create pytorch datasets and dataloader below

# REMEMBER: use torch.from_numpy before creating the dataset! Refer to the NN lecture before for examples

dataset = Torch_Dataset(torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float))

dataloader = DataLoader(dataset, shuffle=True, batch_size=64, num_workers=4)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
# Lets get this model!

# for your output, it will be one node, that outputs the predicted value. What would the output activation function be?

# YOUR CODE HERE

inputSize =  9         # how many classes of input

hiddenSize = 300        # Number of units in the middle

numClasses = 1         

numEpochs = 100         # How many training cycles

learningRate = .001     # Learning rate



class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size) 

        self.fc3 = nn.Linear(hidden_size, hidden_size) 

        self.fc4 = nn.Linear(hidden_size, hidden_size) 

        self.fc5 = nn.Linear(hidden_size, num_classes)  

    

    def forward(self, x):

        x = F.sigmoid(self.fc1(x))

        x = F.sigmoid(self.fc2(x))

        x = F.sigmoid(self.fc3(x))

        x = F.sigmoid(self.fc4(x))

        return self.fc5(x)



model = NeuralNet(inputSize, hiddenSize, numClasses)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = learningRate)

model.to(device)

print(model)
for e in range(numEpochs):

    running_loss = 0.0

    for i, batch in enumerate(dataloader):

        model.train()

        optimizer.zero_grad()

        

        inputs, targets = batch

        inputs, targets = inputs.to(device), targets.to(device)

        

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

        print("Epoch: {}/{}, Batch: {}/{}: train_loss: {}".format(e+1, numEpochs, i, len(dataloader), running_loss/(i+1)))

    print("Epoch: {}/{}, train_loss: {}".format(e+1, numEpochs, running_loss/len(dataloader)))