import torch

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.options.display.max_rows = 10

pd.options.display.float_format = "{:.1f}".format



import warnings

warnings.filterwarnings("ignore")



torch.manual_seed(1)
df = pd.read_csv('/kaggle/input/imdb1000/imdb_data.csv', sep='\t')

df = df.rename(columns={'User Votes': 'Votes',

                        'Imdb Rating': 'Rating',

                       'Gross(in Million Dollars)': 'Earnings',

                       'Runtime(Minutes)' : 'Runtime'})

df.head()
df.describe()
#Correlation between columns to identify best feature for training a model

df.corr()
dataframe = df[['Votes', 'Rating']]

#It is very important to normalise the input features in a proper range

#It helps in avoiding very large calculations

dataframe['Votes'] = dataframe['Votes'] / 1000000

dataframe.head()
#Checking if there's any null values in the dataset

dataframe[dataframe.Votes.isnull()]
plt.figure(figsize=(8,6))

plt.title("Analysis of data points Votes Vs Rating")

sns.scatterplot(x=dataframe.Votes, y=dataframe.Rating)

plt.xlabel('User Votes')

plt.ylabel('IMDB Rating')

plt.show()
from torch.utils.data import Dataset, DataLoader
# x = torch.arange(-4, 4, 0.1).view(-1,1)

# f = -2*x + 3

# df = pd.DataFrame({'X' : x.view(1,-1).numpy()[0], 'Y' : f.view(1,-1).numpy()[0]})

# df.shape[0]
class ratingData(Dataset):

    def __init__(self, df, transform=None):

        self.data = df

        self.transform = transform

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        sample = self.data.iloc[idx, 0], self.data.iloc[idx, 1]

        if self.transform:

            sample = self.transform(sample)

        return sample
dataset = ratingData(dataframe)
from torch import nn, optim
class LinearRegression(nn.Module):

    

    #Constructor for defining the model

    def __init__(self, inp_size, out_size):

        super(LinearRegression, self).__init__()

        self.linear = nn.Linear(inp_size, out_size)

        

    #Prediction/Forward Pass

    def forward(self, x):

        pred = self.linear(x)

        return pred
def train(model, dataloader, epochs, lr):

    LOSS = []

    #Defining Mean Squared Error as loss function

    criterion = nn.MSELoss()

    #Defining optimizer as Stochastic Gradient Descent Optimizer

    optimizer = optim.SGD(model.parameters(), lr=lr)

    

    for epoch in range(epochs):

        print(" Epoch :: ", epoch)

        epochloss = []

        for x, y in dataloader:

            #Making predictions

            pred = model(x.view(-1,1))



            #Claculating loss

            loss = criterion(pred, y.float())

            epochloss.append(loss)



            #Clears the gradients of all optimized tensors

            optimizer.zero_grad()



            #Calculate gradient for loss

            loss.backward()



            #To update the learnable parameters (weight and bias)

            optimizer.step()

        LOSS.append(torch.mean(torch.tensor(epochloss)))

        print("Total Losses :: ",torch.mean(torch.tensor(epochloss)))

    return LOSS
#Hyper-parameters

LOSS = []

epochs = 15

batch_size = len(dataset)

learning_rate = 0.1

#Initialising Model

model = LinearRegression(1,1)

#Creating dataloader to load datasets

dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

#Training the model

LOSS = train(model=model, dataloader=dataloader, epochs=epochs, lr=learning_rate)



print(list(model.parameters()))
weight = model.linear.weight[0].item()

bias = model.linear.bias[0].item()

print(weight)

print(bias)

predictions = weight * dataframe.Votes + bias



plt.figure(figsize=(8,6))

plt.title("Analysis of trained model and data points")

sns.scatterplot(x=dataframe.Votes, y=dataframe.Rating)

sns.lineplot(x=dataframe.Votes, y=predictions, color='red')

plt.xlabel('User Votes')

plt.ylabel('IMDB Rating')

plt.show()
plt.figure(figsize=(8,6))

plt.plot(LOSS, label='BGD')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()