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
dataset = dataset.drop(["HDI for year","country-year"], axis=1)

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
plt.hist(np.log(dataset["gdp_per_capita ($)"]), 100) 

plt.show()
dataset["country"] = dataset["country"]/100

dataset["year"] = (dataset["year"] - 1985)/31

dataset["age"] = dataset["age"]/ 5

dataset["suicides_no"] = dataset["suicides_no"].map(lambda x: np.log(x) if x != 0 else x)

dataset["suicides_no"] = dataset["suicides_no"] / 10.014089

dataset["population"] = np.log(dataset["population"])

dataset["population"] = (dataset["population"] - 5.627621) / (17.595263 - 5.627621)

dataset[" gdp_for_year ($) "] = np.log(dataset[" gdp_for_year ($) "])

dataset[" gdp_for_year ($) "] = (dataset[" gdp_for_year ($) "] - 17.663947) / (30.528077 - 17.663947)

dataset["gdp_per_capita ($)"] = np.log(dataset["gdp_per_capita ($)"])

dataset["gdp_per_capita ($)"] = (dataset["gdp_per_capita ($)"] - 5.525453) / (11.746827 - 5.525453)

dataset["generation"] = dataset["generation"] / 5
dataset.describe()
X, Y = dataset.drop("suicides/100k pop", axis=1).values, dataset["suicides/100k pop"].values

Y = np.expand_dims(Y, axis=1)

X = X.astype("float64")

Y = Y.astype("float64")
# Split data here using train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.3)
print("X shape: {}, Y shape: {}".format(X.shape, Y.shape))
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
# Lets get this model!

# for your output, it will be one node, that outputs the predicted value. What would the output activation function be?

Train_Data = Torch_Dataset(torch.tensor(X_train).float(),torch.tensor(Y_train).float())

Val_Data = Torch_Dataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float())



datasets = {"Train":Train_Data, "Validation":Val_Data}
dataloaders = {x: DataLoader(datasets[x], batch_size=64, shuffle=True, num_workers=0)

              for x in ["Train","Validation"]}
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fc3 = nn.Linear(hidden_size, output_size)

        

    def forward(self, x):

        x = torch.sigmoid(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))

        return self.fc3(x)
def run_epoch(model, dataloaders, device, phase):

    

    running_loss = 0.0

    running_corrects = 0

    

    if phase == "True":

        model.train()

    else:

        model.eval()

        

    for i, (inputs, targets) in enumerate(dataloaders[phase]):

        

        inputs = inputs.to(device)

        targets = targets.to(device)

        

        optimizer.zero_grad()

        

        with torch.set_grad_enabled(phase == "Train"):

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            

            if phase == "Train":

                loss.backward()

                optimizer.step()

                

        preds = torch.round(outputs)

        

        running_loss += loss.item()

        running_corrects += torch.sum(abs(preds - targets) < 0.05 * targets)

        

    epoch_loss = running_loss / datasets[phase].__len__()

    epoch_acc = running_corrects.float() / datasets[phase].__len__()

    

    return epoch_loss, epoch_acc

        
def train(model, criterion, optimizer, num_epochs, dataloaders, device, print_seperation = 1):

    start = time.time()

    

    best_model_wts = model.state_dict()

    best_acc = 0.0

    

    print("| Epoch\t| Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t|")

    print("-" * 73)

    

    for epoch in range(num_epochs):

        train_loss, train_acc = run_epoch(model, dataloaders, device, "Train")

        val_loss, val_acc = run_epoch(model, dataloaders, device, "Validation")

        if (epoch+1) % print_seperation == 0:

            print("| {}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t|".format(epoch + 1, train_loss, train_acc, val_loss, val_acc))

        

        if val_acc > best_acc:

            best_acc = val_acc

            best_model_wts = model.state_dict()

            

    total_time = time.time() - start

    

    print("-" * 74)

    print("Training complete in {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))

    print("Best validation accuracy: {:.4f}".format(best_acc))

    

    model.load_state_dict(best_model_wts)

    return model
#define model variables

input_size = 9

hidden_size = 7

output_size = 1

num_epochs = 1000

learning_rate = 0.01

printing_separation = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

model.to(device)

print(model)
model = train(model, criterion, optimizer, num_epochs, dataloaders, device, printing_separation)