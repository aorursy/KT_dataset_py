import os

import pandas as pd

import numpy as np

import torch.nn as nn

import torch

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import seaborn as sns

from scipy.stats import pearsonr, spearmanr

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error 

from math import sqrt

import time

import datetime

from tqdm import tqdm, trange
DATASET_PATH = '../input/demand-forecasting-kernels-only/' 

# DATASET_PATH = "datasets/store_item_demand/"

df = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"))

df.head(10)
s = lambda x: [item.split("-") for item in x]

dates = s(df.date.values)
df["year"] = [int(item[0]) for item in dates]

df["month"] = [int(item[1]) for item in dates]

df["day"] = [int(item[2]) for item in dates]
# df.drop(["date"], axis=1, inplace=True)

# df.head(10)
sns.pairplot(df[:200000], vars=["month", "year", "sales"])
sales = np.array(df["sales"].values).reshape(-1, 1)

# sales_scaler = StandardScaler()

sales_scaler = MinMaxScaler(feature_range=(-1, 1))

scaled_sales = sales_scaler.fit_transform(sales)
cols = ['store', 'item', 'year', 'month', 'day']

features = df[cols].values

# features_scaler = StandardScaler(with_mean=False)

features_scaler = MinMaxScaler(feature_range=(-1, 1))

scaled_features = features_scaler.fit_transform(features)
x = []

y = []

timesteps = 30

for index in range(len(scaled_features) - timesteps):

  x.append(scaled_features[index:index+timesteps])

  y.append(scaled_sales[timesteps])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
x_train_tensor = torch.tensor(x_train).float()

y_train_tensor = torch.tensor(y_train).float()

x_test_tensor = torch.tensor(x_test).float()

y_test_tensor = torch.tensor(y_test).float()
batch_size = 128

train_data = TensorDataset(x_train_tensor, y_train_tensor)

sampler = SequentialSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=batch_size)



valid_data = TensorDataset(x_test_tensor, y_test_tensor)

sampler = SequentialSampler(valid_data)

valid_dataloader = DataLoader(valid_data, sampler=sampler, batch_size=batch_size)

class LSTMSale(nn.Module):

  def __init__(self, num_features, num_layers, num_hidden_dim, num_output, device):

    super(LSTMSale, self).__init__()

    self.num_layers = num_layers

    self.hidden_dim = num_hidden_dim

    self.device = device

    self.lstm = nn.LSTM(input_size=num_features, hidden_size=num_hidden_dim, num_layers=num_layers, batch_first=True)

    self.fc = nn.Linear(in_features = num_hidden_dim, out_features=num_output)

  

  def forward(self, input):

    h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).requires_grad_()

    c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).requires_grad_()

    h0 = h0.to(self.device)

    c0 = c0.to(self.device)

    logits, (hn, cn) = self.lstm(input, (h0.detach(), c0.detach()))

    output = self.fc(logits[:, -1])

    return output
device = torch.device("cuda:0")

model = LSTMSale(5, 2, 32, 1, device)

optimizer = optim.AdamW(model.parameters(), lr=0.01)

loss_func = nn.MSELoss()

model.cuda()
train_loss_list = []

valid_loss_list = []

valid_rmse = []

epochs = 50

for ep in trange(epochs, desc="Epochs==>"):

    model.train()

    train_loss = 0.0

    valid_loss = 0.0

    t0 = time.time()



    for steps, batch in enumerate(train_dataloader):

        if steps % 10000 == 0 and not steps == 0:

            elapsed_rounded = int(round((time.time() - t0)))

            elapsed = str(datetime.timedelta(seconds=elapsed_rounded))

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(steps, len(train_dataloader), elapsed))

        batch = [t.to(device) for t in batch]

        b_features, b_sales = batch

        pred_sales = model(b_features)

        

        loss = loss_func(pred_sales.view(-1), b_sales.view(-1))

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    train_loss_list.append(train_loss/len(train_dataloader))

    print("\nEpoch {} Train Loss {}".format(ep, train_loss/len(train_dataloader)))



    model.eval()

    true_sales = []

    predicted_sales = []

    for step, batch in enumerate(valid_dataloader):

        batch = [t.to(device) for t in batch]

        b_features, b_sales = batch

        with torch.no_grad():

            p_sales = model(b_features)

        loss = loss_func(p_sales.view(-1), b_sales.view(-1))

        valid_loss += loss.item()

        true_sales.extend(b_sales.detach().cpu().numpy())

        predicted_sales.extend(p_sales.to("cpu").numpy())

    valid_loss_list.append(valid_loss/len(valid_dataloader))

    print("\nEpoch {} Valid Loss {}".format(ep, valid_loss/len(valid_dataloader)))

    rmse = sqrt(mean_squared_error(true_sales, predicted_sales))

    valid_rmse.append(rmse)
import matplotlib.pyplot as plt



plt.plot(train_loss_list, label="Train Loss")

plt.plot(train_loss_list, label="Valid Loss")

plt.plot(valid_rmse, label="Valid RMSE")

plt.legend()

plt.show

# def smape(A, F):

#     return 1/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# model.eval()

# with torch.no_grad():

#   pred_sales = model(x_test_tensor)



# true_sales = y_test_tensor.detach().cpu().numpy()

# pred_sales = pred_sales.to("cpu").numpy()



# rmse_result = sqrt(mean_squared_error(true_sales, pred_sales))

# smape_result = smape(true_sales, pred_sales)



# rmse_result, smape_result
df_test = pd.read_csv(os.path.join(DATASET_PATH, "test.csv"))

s = lambda x: [item.split("-") for item in x]

dates = s(df_test.date.values)

df_test["year"] = [int(item[0]) for item in dates]

df_test["month"] = [int(item[1]) for item in dates]

df_test["day"] = [int(item[2]) for item in dates]

df_test.head(10)
df_test_concat = df[df_test.columns[1:]].tail(30)

df_test_concat["id"] = 0

df_test_concat = df_test_concat.append(df_test, ignore_index=True)

df_test_concat.head(40)
cols = ['store', 'item', 'year', 'month', 'day']

test_features = df_test_concat[cols].values

# features_scaler = StandardScaler(with_mean=False)

# features_scaler = MinMaxScaler(feature_range=(-1, 1))

scaled_test_features = features_scaler.transform(test_features)
scaled_test_features[0], test_features[0]
len(scaled_test_features), len(df_test)
x_test = []

timesteps = 30

for index in range(len(scaled_test_features) - timesteps):

  x_test.append(scaled_features[index:index+timesteps])

x_test_tensor = torch.tensor(x_test).float()
x_test_tensor.shape
model.eval()

with torch.no_grad():

    x_test_tensor = x_test_tensor.to(device)

    pred_sales = model(x_test_tensor)



pred_sales = pred_sales.to("cpu").numpy()



predictions = {"id": df_test["id"], "sales": sales_scaler.inverse_transform(pred_sales).reshape(-1).astype(int)}



prediction_df = pd.DataFrame.from_dict(predictions)

prediction_df.head()

prediction_df.to_csv("submission.csv", index=False)



# prediction_df.head()