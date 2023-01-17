import numpy as np 

import pandas as pd
# parse the train and test file

# 

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.shape

# 显示训练集的shape
test_df.shape

# 显示测试集的shape
train_df.columns

# 显示header
test_df.columns
train_df.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]

# 显示部分数据
# 将训练集和测试集特征79个特征连接起来

# 去除id的列，以及训练集的saleprice列

all_feature = pd.concat([train_df.iloc[:, 1:-1], test_df.iloc[:, 1:]], keys=["train", "test"])
# 预处理数据

numeric_features = all_feature.dtypes[all_feature.dtypes != 'object'].index

numeric_features, all_feature.dtypes

all_feature[numeric_features] = all_feature[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_feature[numeric_features] = all_feature[numeric_features].fillna(0)
all_feature.shape
all_feature = pd.get_dummies(all_feature, dummy_na=True)
all_feature.shape
all_feature.columns
# 特征的index是multiIndex所以需要

all_feature.index.levels[0], all_feature.index
all_feature.index.levels[1]
all_feature.index.labels[0]
all_feature.index.labels[1],all_feature.index.names
all_feature.loc['train', slice(None), :].head()
train_feature = np.array(all_feature.loc['train',].values)
train_feature.size, train_feature.shape


test_feature = np.array(all_feature.loc['test',].values)
# 将label转换成数据输入格式[shape * 1]

train_labels = np.array(train_df.SalePrice.values).reshape((-1,1))
import torch
# 固定种子, 以便重现结果

seed = 2018

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False
torch.cuda.is_available()

# pytorch 来训练一个
torch.cuda.get_device_name()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.current_device()

# 返会当前的设备的索引
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 导入nn module

import torch.nn as nn

from torch.utils.data import DataLoader

import torch.utils.data as data

import torch.nn.functional as F
# 定义一个线性线性回归的网络



class LinearRegression(nn.Module):

    def __init__(self):

        super(LinearRegression, self).__init__()

        self.fc1 = nn.Linear(331, 496)

        self.fc2 = nn.Linear(496, 248)

        self.fc3 = nn.Linear(248, 124)

        self.fc4 = nn.Linear(124, 32)

        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x)).clamp(min=0)

        x = F.dropout(x, p=0.1)

        x = F.relu(self.fc2(x)).clamp(min=0)

        x = F.dropout(x, p=0.1)

        x = F.relu(self.fc3(x)).clamp(min=0)

#         x = F.dropout(x, p=0.1)

        x = F.relu(self.fc4(x)).clamp(min=0)

#         x = F.dropout(x, p=0.1)

        x = F.relu(self.fc5(x)).clamp(min=0)

        return x
# 实例化后使用.to方法将网络移动到GPU

model = LinearRegression().to(DEVICE)
# 查看model

model
import math 

# 定义数据集的读取

class MyData(data.Dataset):

    def __init__(self, feature, label):

        self.feature = feature

        self.label = label



    def __len__(self):

        return len(self.feature)



    def __getitem__(self, idx):

        return self.feature[idx], self.label[idx]



# data_loader = DataLoader(MyData(train_feature, label), batch_size=45, shuffle=True)

# K折交叉验证数据集的获得

def get_k_fold_data(k, i, X, y):

    assert k > 1

    fold_size = X.shape[0] // k

    X_train, y_train = None, None

    for j in range(k):

        idx = slice(j * fold_size, (j + 1) * fold_size)

        X_part, y_part = X[idx, :], y[idx]

        if j == i:

            X_valid, y_valid = X_part, y_part

        elif X_train is None:

            X_train, y_train = X_part, y_part

        else:

            X_train = np.concatenate((X_train, X_part), axis=0)

            y_train = np.concatenate((y_train, y_part), axis=0)

    return X_train, y_train, X_valid, y_valid



class RMSELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.mse = nn.MSELoss()

        

    def forward(self,out,label):

        return torch.sqrt(self.mse(out.float().log(), label.float().log()))



def train(net, train_features, train_labels, test_features, test_labels, 

          num_epochs, learning_rate, weight_decay, batch_size):

    train_ls, test_ls = [], []

    train_dataset = MyData(torch.from_numpy(train_features), torch.from_numpy(train_labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):

        loss_lst = []

        model.train() # set up the train model

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(DEVICE), target.to(DEVICE)

            output = net(data.float())

            # loss = torch.sqrt(F.mse_loss(output.float().log(), target.float().log()))

            criterion  = RMSELoss()

            loss = criterion(output, target)

            loss_lst.append(loss.item())

            optimizer.zero_grad() # optimizer 0

            loss.backward() # back propragation

            optimizer.step() # update the paramters 

        print('Train Epoch: {} \tLoss: {}'.format(epoch, sum(loss_lst)/batch_idx))

        train_ls.append(sum(loss_lst)/batch_idx)

        

        if test_labels is not None:

            # val

            val_dataset = MyData(torch.from_numpy(test_features), torch.from_numpy(test_labels))

            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            model.eval()

            val_loss = 0

            with torch.no_grad():

                for data, target in val_loader:

                    data, target = data.to(DEVICE), target.to(DEVICE)

                    output = net(data.float())

                    val_loss += F.mse_loss(output.float().log(), target.float().log(),reduction='sum').item() # 将一批的损失相加

            val_loss = math.sqrt(val_loss/len(val_loader.dataset))      

            print('\nTest set: Average loss: {:.4f}\n'.format(val_loss))

            test_ls.append(val_loss)

    return train_ls, test_ls

    

    
import matplotlib.pyplot as plt
# 绘图

    




# 权重和bias的初始化

for m in model.modules():

    if isinstance(m, nn.Linear):

        nn.init.xavier_uniform_(m.weight)

        m.bias.data.fill_(0.01)



def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):

    train_l_sum, valid_l_sum = 0, 0

    plt.figure(figsize=(15,8))

    for i in range(k):

#         print('fold {}'.format(str(i)))

        model = LinearRegression().to(DEVICE)

        data = get_k_fold_data(k, i, X_train, y_train)

        trainloss, valloss = train(model, *data, num_epochs, learning_rate, weight_decay, batch_size)

        plt.subplot(5,1,i+1)

        plt.plot(trainloss, 'g-')

        plt.plot(valloss, 'r-')

        train_l_sum += trainloss[-1]

        valid_l_sum += valloss[-1]

        print('fold %d, train rmse %f, valid rmse %f'% (i, trainloss[-1], valloss[-1]))

       

    plt.show()

#     return train_l_sum / k, valid_l_sum / k
k, num_epochs, lr, weight_decay, batch_size =  5, 1000, 0.001, 0, 64 # hyper paramters

k_fold(k, train_feature, train_labels, num_epochs, lr, weight_decay, batch_size)
# pip install nvidia-ml-py3
import pynvml

pynvml.nvmlInit()

# 这里的0是GPU id

handle = pynvml.nvmlDeviceGetHandleByIndex(0)

meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(meminfo.used/1024**3)
print(meminfo.total/1024**3)
type(meminfo.total)
def train_and_pred(train_features, test_features, train_labels, test_data, 

                   num_epochs, lr, weight_decay, batch_size):

    net = LinearRegression().to(DEVICE)

    # 权重和bias的初始化

    for m in net.modules():

        if isinstance(m, (nn.Conv2d, nn.Linear)):

            nn.init.xavier_uniform_(m.weight)

            m.bias.data.fill_(0.01)

    train(net, train_features, train_labels, None, None,num_epochs, lr, weight_decay, batch_size)

    test_features = torch.from_numpy(test_features).to(DEVICE)

    with torch.no_grad():

        net.eval()

        preds = net(test_features.float())

    preds = preds.cpu().numpy()

#     print(preds.shape)

    test_data['SalePrice'] = pd.Series(preds.reshape(-1))

    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)

    submission.to_csv('submission.csv', index=False)
torch.__version__

k, num_epochs, lr, weight_decay, batch_size =  5, 500, 0.001, 0.0005, 64 # hyper paramters
train_and_pred(train_feature, test_feature, train_labels, test_df, num_epochs, lr, weight_decay, batch_size)