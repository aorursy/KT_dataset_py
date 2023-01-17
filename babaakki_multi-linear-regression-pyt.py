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

df.corr()
dataframe = df[['Votes', 'Earnings', 'Rating']]
#It is very important to normalise the input features in a proper range
#It helps in avoiding very large calculations
dataframe['Votes'] = dataframe['Votes'] / 1000000
dataframe['Earnings'] = dataframe['Earnings'] / 100
dataframe.describe()
#There are 73 (1000 - 927) rows with NaN/nan values
#Drop those rows
dataframe.dropna(inplace=True)
#Empty dataframe confirms abscence of rows with nan/NaN
dataframe[dataframe.Earnings.isnull()]
from torch.utils.data import Dataset, DataLoader, random_split
class ratingData(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data.iloc[idx, [0,1]].values), self.data.iloc[idx, 2]
        if self.transform:
            sample = self.transform(sample)
        return sample
dataset = ratingData(dataframe)
len(dataset)
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
#Since we are passing multiple inputs (Votes and Earnings), input size has been changed to 2
model = LinearRegression(2, 1)
list(model.parameters())
model.state_dict()
dataloader = DataLoader(dataset, shuffle=True, batch_size=len(dataset))
lr = 0.1
epochs = 50
criterion = nn.MSELoss()
LOSS = []
optimizer = optim.SGD(model.parameters(), lr=lr)
for epoch in range(epochs):
    epochloss = []
    for x, y in dataloader:
        pred = model(x.float())
        loss = criterion(pred, y)
        epochloss.append(loss)
        #Setting gradients to 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch :: {}, Loss :: {}".format(epoch, torch.mean(torch.tensor(epochloss))))
    LOSS.append(torch.mean(torch.tensor(epochloss)).item())
plt.figure(figsize=(8, 6))
plt.plot(LOSS, label="LOSS")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
