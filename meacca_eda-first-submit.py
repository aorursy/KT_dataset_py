import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')



train.sort_values('sig_id', inplace=True)

train_scored.sort_values('sig_id', inplace=True)



test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
sum(train.isna().sum())
train.head()
cat_features = ['cp_type', 'cp_time', 'cp_dose']

gen_features = [column for column in train.columns if column.startswith('g-')]

cell_features = [column for column in train.columns if column.startswith('c-')]
print('Число категориальных признаков: ', len(cat_features))

print('Число генных признаков: ', len(gen_features))

print('Число клеточных признаков: ', len(cell_features))
train.set_index('sig_id', inplace=True)

train_scored.set_index('sig_id', inplace=True)

test.set_index('sig_id', inplace=True)
for feature in cat_features:

    train[feature] = train[feature].astype('category')

    test[feature] = test[feature].astype('category')
index = list(train['cp_type'].value_counts().index)

height = list(train['cp_type'].value_counts())
fig, ax = plt.subplots(1, 3, figsize=(15, 5))



for i, feature in enumerate(cat_features, 1):

    plt.subplot(1, 3, i)

    plt.title(feature)

    index = list(train[feature].astype('str').value_counts().index)

    height = list(train[feature].value_counts())

    plt.bar(x=index, height=height, width=0.5)

plt.show()
plt.figure(figsize=(10, 7))

plt.hist(train_scored.mean(axis=0) * 100, bins=20)

plt.xlabel('Percent of 1', fontsize=15)

plt.show()
const_target = {}



for feature in cat_features:

    const_target[feature] = {}

    for value in train[feature].unique():

        sum_target = train_scored.loc[train[train[feature] == value].index].sum(axis=0)

        const_target[feature][value] = -1 * np.ones_like(sum_target)

        const_target[feature][value][sum_target == 0] = 0
for feature in const_target:

    print(feature)

    for value in const_target[feature]:

        print(value)

        print('Const zeros target: ', sum(const_target[feature][value] == 0))

    print()
print(np.where(const_target['cp_time'][24] == 0)[0])

print(np.where(const_target['cp_time'][72] == 0)[0])

print(np.where(const_target['cp_dose']['D2'] == 0)[0])
print(sum(train_scored[train_scored.columns[34]]))

print(sum(train_scored[train_scored.columns[82]]))
train[np.random.choice(gen_features, 9, replace=False)].hist(figsize=(15, 15), bins=20)

plt.show()
train[np.random.choice(cell_features, 9, replace=False)].hist(figsize=(15, 15), bins=20)

plt.show()
plt.figure(figsize=(12, 8))

train[gen_features].std().hist(density=True, bins=20)

train[cell_features].std().hist(density=True, bins=20)

plt.title('Распределение std по признакам на трейне', fontsize=15)

plt.legend(['Генные признаки', 'клеточные признаки'], fontsize=15)

plt.show()
plt.figure(figsize=(12, 8))

test[gen_features].std().hist(density=True, bins=20)

test[cell_features].std().hist(density=True, bins=20)

plt.title('Распределение std по признакам на тесте', fontsize=15)

plt.legend(['Генные признаки', 'клеточные признаки'], fontsize=15)

plt.show()
plt.figure(figsize=(12, 8))

train[gen_features].mean().hist(density=True, bins=20)

train[cell_features].mean().hist(density=True, bins=20)

plt.title('Распределение mean по признакам на трейне', fontsize=15)

plt.legend(['Генные признаки', 'клеточные признаки'], fontsize=15)

plt.show()
plt.figure(figsize=(12, 8))

test[gen_features].mean().hist(density=True, bins=20)

test[cell_features].mean().hist(density=True, bins=20)

plt.title('Распределение mean по признакам на тесте', fontsize=15)

plt.legend(['Генные признаки', 'клеточные признаки'], fontsize=15)

plt.show()
train.reset_index(inplace=True)

test.reset_index(inplace=True)



var_threshold = VarianceThreshold(threshold=0.7)



real_data = train.append(test)

real_data_transformed = var_threshold.fit_transform(real_data.iloc[:, 4:])



train_real_transformed = real_data_transformed[ :train.shape[0]]

test_real_transformed = real_data_transformed[train.shape[0]: ]





train = pd.DataFrame(train[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),

                     columns=['sig_id','cp_type','cp_time','cp_dose'])

train = pd.concat([train, pd.DataFrame(train_real_transformed)], axis=1)





test = pd.DataFrame(test[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                    columns=['sig_id','cp_type','cp_time','cp_dose'])

test = pd.concat([test, pd.DataFrame(test_real_transformed)], axis=1)
train.set_index('sig_id', inplace=True)

need_indexes = train[(train['cp_type'] != 'ctl_vehicle')].index
# в предсказании заполним нулями

train = train.loc[need_indexes].copy()

train_scored = train_scored.loc[need_indexes].copy()



test = test[test['cp_type'] != 'ctl_vehicle'].copy()



train.reset_index(inplace=True)

train_scored.reset_index(inplace=True)

test.reset_index(drop=True, inplace=True)
train.drop('cp_type', axis=1, inplace=True)

test.drop('cp_type', axis=1, inplace=True)
cp_time_item2index = {elem: index for index, elem in enumerate(train['cp_time'].unique(), 0)}

cp_dose_item2index = {elem: index for index, elem in enumerate(train['cp_dose'].unique(), 0)}
train['cp_time'] = train['cp_time'].map(cp_time_item2index)

train['cp_dose'] = train['cp_dose'].map(cp_dose_item2index)



test['cp_time'] = test['cp_time'].map(cp_time_item2index)

test['cp_dose'] = test['cp_dose'].map(cp_dose_item2index)
train.head()
class Data(Dataset):

    def __init__(self, data: pd.DataFrame, target: pd.DataFrame):

        self.data = data[data.columns[1:]]

        self.y = target[target.columns[1:]]

        

    def __getitem__(self, index):

        return torch.tensor(self.data.iloc[index], dtype=torch.float), torch.tensor(self.y.iloc[index], dtype=torch.float)

    

    def __len__(self):

        return len(self.data)

    

    

class TestData(Dataset):

    def __init__(self, data: pd.DataFrame):

        self.data = data[data.columns[1:]]

        

    def __getitem__(self, index):

        return torch.tensor(self.data.iloc[index], dtype=torch.float)

    

    def __len__(self):

        return len(self.data)
train_indexes, test_indexes = train_test_split(np.arange(len(train)), train_size=0.8, random_state=42)
train_data = Data(train.loc[train_indexes], train_scored.loc[train_indexes])

valid_data = Data(train.loc[test_indexes], train_scored.loc[test_indexes])
device = ('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=1)

valid_loader = DataLoader(valid_data, batch_size=16, num_workers=1)
class DummyModel(nn.Module):

    def __init__(self, num_features, hidden_size, num_classes):

        super().__init__()

        self.linear_in = nn.Linear(num_features, hidden_size)

        self.norm = nn.BatchNorm1d(hidden_size)

        self.drop = nn.Dropout(0.2)

        self.linear_out = nn.Linear(hidden_size, num_classes)

        

    def forward(self, x):

        x = self.linear_in(x)

        x = self.norm(x)

        x = F.relu(self.drop(x))

        x = self.linear_out(x)

        return x
model = DummyModel(train.shape[1] - 1, 512, train_scored.shape[1] - 1)

model.to(device)
loss_func = nn.BCEWithLogitsLoss(reduction='none')

loss_func.to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.95, step_size=1)

#scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=1e-2, total_steps=30)
num_epochs = 10



for i in range(num_epochs):

    epoch_loss = torch.zeros(train_scored.shape[1] - 1)

    for batch_idx, batch in enumerate(train_loader, 1):

        output = model(batch[0].to(device))

        target = batch[1].to(device)

        loss = loss_func(output, target)

        loss.mean().backward()



        optimizer.step()

        optimizer.zero_grad()

    #scheduler.step()

    

    model.eval()

    print(f"EPOCH {i+1}")

    train_loss = torch.zeros(train_scored.shape[1] - 1)

    with torch.no_grad():

        for batch in train_loader:

            output = model(batch[0].to(device))

            target = batch[1].to(device)

            loss = loss_func(output, target)

            batch_loss = loss.cpu().data.mean(dim=0)

            train_loss += batch_loss

        print("Train loss", float((train_loss / len(train_loader)).mean()))

    

    log_loss = torch.zeros(train_scored.shape[1] - 1)

    test_loss = 0

    with torch.no_grad():

        for batch in valid_loader:

            output = model(batch[0].to(device))

            target = batch[1].to(device)

            loss = loss_func(output, target)

            

            batch_loss = loss.cpu().data.mean(dim=0)

            test_loss += float(loss.cpu().data.mean())

            log_loss += batch_loss

        print("Log loss", float((log_loss / len(valid_loader)).mean()))

        print("Test loss", test_loss / len(valid_loader))

            

    

    model.train()
sub_ex = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
test_data = TestData(test)

test_loader = DataLoader(test_data, batch_size=16, num_workers=1)
model.eval()

preds = []

with torch.no_grad():

    for batch in test_loader:

        preds.append(model(batch))

        

preds = torch.cat(preds, dim=0)

preds = torch.sigmoid(preds)
res = pd.DataFrame(test['sig_id'])
res = pd.concat((res, pd.DataFrame(preds.cpu().numpy())), axis=1)
res = sub_ex[['sig_id']].merge(res, on='sig_id', how='left')

res = res.fillna(1e-5)

res.columns = sub_ex.columns
res.to_csv('/kaggle/working/submission.csv', index=False)