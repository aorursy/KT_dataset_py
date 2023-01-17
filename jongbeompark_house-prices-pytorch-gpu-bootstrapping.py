import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame



import matplotlib.pyplot as plt

import seaborn as sns



import math
raw_data_train = pd.read_csv('../input/train.csv')

raw_data_test = pd.read_csv('../input/test.csv')
raw_data = pd.concat([raw_data_train, raw_data_test], axis=0, sort=False).reset_index(drop=True)
raw_data.head()
raw_data.tail()
numeric_columns = []

numeric_columns.extend(list(raw_data.dtypes[raw_data.dtypes == np.int64].index))

numeric_columns.extend(list(raw_data.dtypes[raw_data.dtypes == np.float64].index))



numeric_columns.remove('SalePrice')

numeric_columns.append('SalePrice')

numeric_columns.remove('Id')
non_numeric_columns = [col for col in list(raw_data.columns) if col not in numeric_columns]

non_numeric_columns.remove('Id')
for col in numeric_columns:

    raw_data[col] = raw_data[col].fillna(0)

    

for col in non_numeric_columns:

    raw_data[col] = raw_data[col].fillna('N/A')
mapping_table = dict()



for col in non_numeric_columns:

    curr_mapping_table = dict()

    curr_mapping_table['N/A'] = 0

    

    unique_values = pd.unique(raw_data[col])

    idx = 1

    for inx, v in enumerate(unique_values):

        if not v in curr_mapping_table.keys():

            curr_mapping_table[v] = idx

            idx += 1

        raw_data[col] = raw_data[col].replace(v, curr_mapping_table[v])

    

    mapping_table[col] = curr_mapping_table
means, stds = dict(), dict()



for col in raw_data.columns:

    if col == 'SalePrice':

        continue

    means[col] = raw_data[col].mean()

    stds[col] = raw_data[col].std()
# Finding Info of SalePrice

means['SalePrice'] = raw_data_train['SalePrice'].mean()

stds['SalePrice'] = raw_data_train['SalePrice'].std()
for col in raw_data.columns:

    raw_data[col] = (raw_data[col] - means[col]) / (stds[col])
raw_data.head()
train_data = DataFrame(raw_data, index=raw_data_train.index)
train_data.head()
train_data.tail()
def get_bootstrapped_dataset(original_data):

    random_index = np.random.randint(original_data.shape[0], size=original_data.shape[0])

    return DataFrame(original_data, index=random_index).reset_index(drop=True)
datasets = []



for __ in range(int(math.sqrt(train_data.shape[0]))):

    datasets.append(get_bootstrapped_dataset(train_data))
x_columns = list(train_data.columns)

x_columns.remove('SalePrice')

x_columns.remove('Id')



y_columns = ['SalePrice']
x_dfs, y_dfs = [], []



for dataset in datasets:

    x_dfs.append(DataFrame(dataset, columns=x_columns))

    y_dfs.append(DataFrame(dataset, columns=y_columns))
import torch

import torch.nn as nn

import torch.optim as optim



import time
epoch = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):

    def __init__(self, D_in, H1, H2, H3, D_out):

        super(Net, self).__init__()

        

        self.linear1 = nn.Linear(D_in, H1)

        self.linear2 = nn.Linear(H1, H2)

        self.linear3 = nn.Linear(H2, H3)

        self.linear4 = nn.Linear(H3, D_out)

        

    def forward(self, x):

        y_pred = self.linear1(x).clamp(min=0)

        y_pred = torch.nn.functional.dropout(y_pred, p=0.2)

        y_pred = self.linear2(y_pred).clamp(min=0)

        y_pred = torch.nn.functional.dropout(y_pred, p=0.2)

        y_pred = self.linear3(y_pred).clamp(min=0)

        y_pred = torch.nn.functional.dropout(y_pred, p=0.2)

        y_pred = self.linear4(y_pred)

        return y_pred
models = []

losses = []
for inx in range(len(datasets)):

    curr_x_df, curr_y_df = x_dfs[inx], y_dfs[inx]

    

    x = torch.from_numpy(curr_x_df.values).to(device).float()

    y = torch.from_numpy(curr_y_df.values).to(device).float()

    

    D_in, H1, H2, H3, D_out = x_dfs[inx].shape[1], 1000, 500, 200, y_dfs[inx].shape[1]

    

    model = Net(D_in, H1, H2, H3, D_out).to(device)

    criterion = nn.MSELoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    

    curr_losses = []

    

    start_time = time.time()



    for t in range(epoch):

        y_pred = model(x)



        loss = criterion(y_pred, y)

        curr_losses.append(loss.item())



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    end_time = time.time()

    print(inx, end_time - start_time)



    models.append(model)

    losses.append(curr_losses)

plt.figure(figsize=(20, 10))

for curr_losses in losses:

    plt.plot(range(len(curr_losses)), curr_losses)

plt.show()
plt.figure(figsize=(20, 10))

for curr_losses in losses:

    plt.plot(range(len(curr_losses[-100:])), curr_losses[-100:])

plt.show()
raw_test = pd.read_csv('../input/test.csv')
for col in numeric_columns:

    if col == 'SalePrice':

        continue

    raw_test[col] = raw_test[col].fillna(0)

    

for col in non_numeric_columns:

    raw_test[col] = raw_test[col].fillna('N/A')

    
for col in non_numeric_columns:

    curr_mapping_table = mapping_table[col]

    for k, v in curr_mapping_table.items():

        raw_test[col] = raw_test[col].replace(k, v)

    
for col in raw_test.columns:

    raw_test[col] = (raw_test[col] - means[col]) / (stds[col])
raw_test.head(10)
test_x = torch.from_numpy(DataFrame(raw_test, columns=x_columns).values).to(device)

test_x = test_x.float()
test_y = models[0](test_x)    

for inx in range(1, len(models)):

    test_y = test_y + models[inx](test_x)



test_y = test_y / len(models)
test_y = test_y.to('cpu')
result = DataFrame(test_y.data.numpy())

result = result.rename(columns={0: 'SalePrice'})
result['Id'] = result.index

result['Id'] = result['Id'] + 1461



result = DataFrame(result, columns=['Id', 'SalePrice'])
result['SalePrice'] = result['SalePrice'] * (stds['SalePrice']) + means['SalePrice']
result.head()
result.to_csv('./submission.csv', columns=['Id', 'SalePrice'], index=False)