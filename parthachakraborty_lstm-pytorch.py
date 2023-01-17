!apt-get -qq install -y graphviz && pip install -q pydot

!pip install torchvision

!pip install torchviz

!pip install -q kaggle

!pip install torchsummary
import numpy as np

from matplotlib import pyplot as plt

import torch

import torch.nn as nn

import torchvision.transforms as transforms

import torchvision.datasets as dests

from torch.utils import data

from torch.autograd import Variable

from torchvision import models

from torchsummary import summary

from torchviz import make_dot

import pandas as pd

import os

from shutil import copyfile

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from PIL import Image

from numpy import asarray

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
dateparse = lambda x: datetime.strptime(x, "%Y-%m-%d")

dateconvert = lambda x: datetime.utcfromtimestamp(x*(1e-9)).strftime('%Y-%m-%d')

INPUT_PATH = "/kaggle/input/daily-historical-stock-prices-1970-2018"

OUTPUT_PATH = "stock"

from_year = '2000'

sequence_dim = 6 # number of look back data to predict one forward
df = pd.read_csv(INPUT_PATH+"/historical_stock_prices.csv",parse_dates=['date'],date_parser=dateparse)
df = df.loc[(df.date>from_year)]
company_list = df['ticker'].unique().tolist()

company_list.sort()

date_list = df['date'].unique().tolist()

date_list = [ dateconvert(x) for  x in date_list]

date_list.sort()
df.set_index(['ticker', 'date'], inplace=True)

df = df.sort_values("date")
highest_data = -1

for item in company_list:

  if  len(df.loc[item]) >= highest_data:

    highest_data = len(df.loc[item])
temp = []

for item in company_list:

  if  len(df.loc[item]) >=highest_data:

    temp.append(item)

company_list = temp

company_list.sort()
row_list = []

column_to_be_predicted = "open"

i = 0

for date in date_list:

    row = [date]

    for company in company_list:

        try:

          row.append(getattr(df.loc[company, date],column_to_be_predicted))

        except KeyError:

          print(company,date)

          row.append(-1)



    row_list.append(tuple(row))

    i += 1

    print((i/len(date_list))*100)
df = pd.DataFrame(data=row_list, columns=["Date"] + company_list)

# if not os.path.exists("/content/drive/My Drive"+"/sanitized.csv"):

#   df = pd.DataFrame(data=row_list, columns=["Date"] + company_list)

#   df.to_csv(path_or_buf="/content/drive/My Drive"+"/sanitized.csv",index=False)

#   df.to_csv(path_or_buf=OUTPUT_PATH+"/sanitized.csv",index=False)

# else:

#   df = pd.read_csv("/content/drive/My Drive"+"/sanitized.csv")
df.head()
number_of_company = 5

full_data = df.iloc[:, 1:number_of_company+2]

train_data, test_data = full_data.iloc[:round(len(full_data)*0.8),1:], full_data.iloc[round(len(full_data)*0.8):,1:]

scaler = MinMaxScaler(feature_range = (0, 1))

scaled_train_data = scaler.fit_transform(train_data)

scaled_test_data = scaler.transform(test_data)
x_train = []

y_train = []

for i in range(sequence_dim, len(scaled_train_data)):

    x_train.append(scaled_train_data[i-sequence_dim:i, :])

    y_train.append(scaled_train_data[i, :])

x_train, y_train = np.array(x_train).astype(float), np.array(y_train).astype(float)

x_train[0][0].dtype
x_test = []

y_test = []

for i in range(sequence_dim, len(scaled_train_data)):

    x_test.append([scaled_train_data[i-sequence_dim:i, :]])

    y_test.append(scaled_train_data[i, :])

x_test, y_test = np.array(x_train).astype(np.float64), np.array(y_train).astype(np.float64)
class StockDataset(data.Dataset):

  'Characterizes a dataset for PyTorch'

  def __init__(self, object_list, labels, transform=None):

        'Initialization'

        self.labels = labels

        self.object_list = object_list

        self.transform = transform



  def __len__(self):

        'Denotes the total number of samples'

        return len(self.object_list)



  def __getitem__(self, index):

        'Generates one sample of data'

        # Select sample

        X = self.object_list[index]



        if self.transform:

            for transform_item in self.transform:

                X = transform_item(X)

        y = self.labels[index]



        return X, y
train_dataset = StockDataset(object_list=x_train, labels= y_train)

test_dataset = StockDataset(object_list=x_test, labels= y_test)
batch_size = 800

n_iters = 12000

num_epochs = int(n_iters / (len(train_dataset)/batch_size))
train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle=False)
class LSTMModel(nn.Module):

    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim):

        super(LSTMModel, self).__init__()

        self.layer_dim = layer_dim

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim,hidden_dim,layer_dim,batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):

        h0 = Variable(torch.zeros(self.layer_dim, x.size(0),self.hidden_dim))

        c0 = Variable(torch.zeros(self.layer_dim, x.size(0),self.hidden_dim))

        out,(hn,cn)= self.lstm(x,(h0,c0)) # hn shape layer_dim, batch_size, hidden_dim out, shape batch_size, seq_dim, hidden_dim

        out = self.fc(out[:,-1,:])

        return out
input_dim = number_of_company

hidden_dim = 100

layer_dim = 2

output_dim = number_of_company

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).float()

criterion = nn.MSELoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
iter_counter = 0

for epoch in range(num_epochs):

    for i, (images,labels) in enumerate(train_loader):

        images= Variable(images)

        labels = Variable(labels)

        optimizer.zero_grad()

        outputs = model(images.float())

        loss = criterion(outputs,labels.float())

        loss.backward()

        optimizer.step()

        iter_counter +=1

        if iter_counter% 500 ==0:

            error = 0

            for images, labels in test_loader:

                images = Variable(images.float())

                outputs = model(images.float())

                error += ((outputs.data - labels.data)**2).mean()

            print("Iteration: {} Loss: {} Error: {}".format(iter_counter, loss, error))

                

            
Flag_first = True

for prices, labels in test_loader:

    prices = Variable(prices.float())

    outputs = model(images.float())

    if Flag_first:

      actual_data = scaler.inverse_transform(labels.data.numpy())

    else:

      np.concatenate(actual_data,scaler.inverse_transform(labels.data.numpy()))

    if Flag_first:

      predicted_data = scaler.inverse_transform(outputs.data.numpy())

    else:

      np.concatenate(actual_data,scaler.inverse_transform(outputs.data.numpy()))
for item in range(number_of_company):



  plt.plot(actual_data.T[0], color = 'blue', label = 'Actual Stock Price')

  plt.plot(predicted_data.T[0], color = 'red', label = 'Predicted Stock Price')

  plt.title('Stock Price Prediction')

  plt.xlabel('Time')

  plt.ylabel('Stock Price')

  plt.legend()

  plt.show()