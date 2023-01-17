import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import torch

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset



from sklearn.model_selection import train_test_split
data_dir = '../input/lish-moa/'



train_features = pd.read_csv(data_dir+'train_features.csv')

test_features = pd.read_csv(data_dir+'test_features.csv')

train_target = pd.read_csv(data_dir+'train_targets_scored.csv')



# train_target
# target_corr = train_target.corr()



# mask = ((target_corr > 0.6).sum(axis=1) - 1) > 0



# plt.figure(figsize=(10,10))

# sns.heatmap(target_corr.loc[mask, mask])



# np.sum(((target_corr > 0.6).sum(axis=1) - 1) * mask)



# train_cols = train_features.columns.values[train_features.dtypes != 'object']

# target_cols = train_target.columns.values[train_target.dtypes != 'object']



# corr = np.zeros((len(train_cols), len(target_cols)))



# for i, train_col in enumerate(train_cols):

#     for j, target_col in enumerate(target_cols):

# #         print(train_features[train_col].dtype,train_target[target_col].dtype )

#         corr[i,j] = train_features[train_col].corr(train_target[target_col])



# corr[np.isnan(corr)] = 0



# plt.figure(figsize=(10,10))

# sns.heatmap(np.abs(corr))



# np.percentile(np.abs(corr), 80)



# np.max(np.abs(corr))



# plt.figure(figsize=(6,6))

# tmp = np.abs(corr).ravel()

# plt.hist(tmp[(tmp < np.percentile(tmp, 99)) & (tmp > 0)], bins=100)

# plt.show()



# n_corrv = (np.abs(corr) > 0.01).sum(axis=1)



# plt.hist(n_corrv, bins=40)



# np.percentile(n_corrv, 80)



# (n_corrv > 98).sum()



# best_features = train_cols[np.argsort(n_corrv)[::-1]][:172]



# best_features
best_features = np.array(['g-392', 'c-65', 'g-100', 'c-9', 'g-50', 'c-79', 'c-98', 'g-37',

       'c-6', 'c-26', 'g-439', 'g-628', 'g-744', 'g-351', 'g-298', 'c-42',

       'g-410', 'g-761', 'c-18', 'g-418', 'g-146', 'c-57', 'c-64',

       'g-322', 'c-48', 'c-38', 'c-82', 'g-63', 'g-534', 'c-92', 'g-186',

       'c-28', 'g-486', 'c-70', 'g-672', 'g-91', 'g-731', 'c-10', 'g-386',

       'g-121', 'g-443', 'g-206', 'g-723', 'c-81', 'c-36', 'c-33', 'g-85',

       'g-235', 'g-406', 'g-683', 'c-52', 'c-62', 'c-63', 'g-365', 'c-21',

       'c-60', 'c-15', 'c-66', 'g-629', 'c-49', 'g-248', 'c-59', 'c-24',

       'c-76', 'g-669', 'g-106', 'g-38', 'g-140', 'c-30', 'c-22', 'g-72',

       'c-25', 'c-23', 'c-8', 'c-83', 'g-489', 'g-369', 'c-47', 'g-158',

       'g-297', 'g-147', 'c-5', 'c-77', 'g-163', 'g-332', 'g-344', 'c-50',

       'g-335', 'c-2', 'g-503', 'g-208', 'g-152', 'c-17', 'c-41', 'g-353',

       'c-34', 'g-664', 'c-96', 'g-228', 'c-67', 'g-569', 'g-750', 'g-30',

       'g-578', 'c-90', 'c-72', 'g-257', 'c-75', 'c-97', 'g-98', 'g-500',

       'c-1', 'g-728', 'c-44', 'g-360', 'c-85', 'g-195', 'c-31', 'c-11',

       'c-40', 'g-135', 'g-65', 'c-95', 'c-80', 'g-261', 'g-590', 'c-54',

       'c-51', 'c-13', 'c-12', 'g-201', 'g-83', 'g-468', 'g-58', 'g-478',

       'g-460', 'g-574', 'c-45', 'c-94', 'c-4', 'g-367', 'c-69', 'g-407',

       'c-73', 'g-349', 'g-155', 'g-113', 'g-350', 'c-91', 'g-546',

       'g-131', 'g-52', 'g-745', 'c-55', 'c-27', 'c-14', 'g-379', 'g-51',

       'g-199', 'g-241', 'g-568', 'g-10', 'c-93', 'g-508', 'c-84', 'c-78',

       'g-433', 'c-20', 'c-39', 'g-7', 'g-177', 'g-185'])
train_features_new = train_features[best_features]



train_target = train_target.drop('sig_id', axis=1)







# train, test, y_train, y_test = train_test_split(train_features_new, train_target,

#                                                 test_size = 0.2, random_state=111)





# train_dataset = TensorDataset(torch.tensor(train.values, dtype=torch.float32),

#                               torch.tensor(y_train.values, dtype=torch.float32))

# test_dataset = TensorDataset(torch.tensor(test.values, dtype=torch.float32),

#                              torch.tensor(y_test.values, dtype=torch.float32))



train_dataset = TensorDataset(torch.tensor(train_features_new.values, dtype=torch.float32),

                              torch.tensor(train_target.values, dtype=torch.float32))

# test_dataset = TensorDataset(torch.tensor(test.values, dtype=torch.float32),

#                              torch.tensor(y_test.values, dtype=torch.float32))



train_dataloader = DataLoader(train_dataset, batch_size=256)
class MyModel(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.in_features = in_features

        self.out_features = out_features



        self.model = nn.Sequential(

            nn.BatchNorm1d(in_features),

            nn.Linear(in_features, 2 * in_features),

            nn.ReLU(),



            nn.BatchNorm1d(2 * in_features),

            nn.Dropout(0.1),

            nn.Linear(2 * in_features, 4 * in_features),

            nn.ReLU(),



            nn.BatchNorm1d(4 * in_features),

            nn.Dropout(0.1),

            nn.Linear(4 * in_features, 2 * in_features),

            nn.ReLU(),



            nn.BatchNorm1d(2 * in_features),

            nn.Linear(2 * in_features, out_features),

            nn.Sigmoid()

        )



    def forward(self, x):

        return self.model(x)
model = MyModel(train.shape[1], y_train.shape[1])

optimizer = torch.optim.Adam(model.parameters()) #SGD(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.05, 10)

criterion = nn.BCELoss()

max_epoch = 20



for epoch in range(max_epoch):

    model.train()

    for i, (x_batch, y_batch) in enumerate(train_dataloader):

        preds = model(x_batch)



        optimizer.zero_grad()

        loss = criterion(preds, y_batch)

        loss.backward()

        optimizer.step()



        if i % 20 == 0:

            print(f'Epoch: {epoch}, train loss: {loss.item():12.5f}')



    model.eval()

    with torch.no_grad():

        train, y_train = train_dataset.tensors

        train_preds = model(train)

        train_loss = criterion(train_preds, y_train).item()

        

#         test, y_test = test_dataset.tensors

#         test_preds = model(test)

#         test_loss = criterion(test_preds, y_test).item()

        print(f'Epoch {epoch} final: train loss: {train_loss}')#, f'test loss: {test_loss}')
test_features_new = test_features[best_features]
model.eval()

with torch.no_grad():

    probs = model(torch.tensor(test_features_new.values, dtype=torch.float32))
sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')
sample_submission
sample_submission.iloc[:, 1:] = probs
sample_submission
sample_submission.to_csv('submission.csv', index=False)