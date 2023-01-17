import torch
import torch.nn as nn   
import pandas as pd
import numpy as np
import warnings
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
path = '/kaggle/input/dataset/'
train_data = pd.read_csv(path + 'used_car_train_20200313.csv', sep=' ')
test_data = pd.read_csv(path + 'used_car_testA_20200313.csv', sep=' ')
train_data.drop(index=train_data.index[train_data['model'].isnull()], inplace=True) # 删除model为None的行
#train_label_ground = train_data['price']
#train_data['price'] = np.log(train_data['price']) no change price
## 输出数据的大小信息
print('train data shape:',train_data.shape)
print('test data shape:',test_data.shape)
train_data['istrain'] = 1
test_data['istrain'] = 0
data = pd.concat([train_data, test_data], ignore_index=True)
data.drop(columns=['name', 'offerType', 'seller'], inplace=True) # 删除意义不大的特征
#data.loc[:, data.columns != 'price']=data.loc[:, data.columns != 'price'].apply(lambda x:x.fillna(x.value_counts().index[0])) # fill NA with most frequent value
# attention: mabe -1 is better
#data.fillna({'bodyType': -1, 'brand': -1, 'gearbox': -1}, inplace=True)
#data['power'][data['power'] > 600] = 0
#data['power'].replace(0, mean_power, inplace=True)
date_features = ['regDate', 'creatDate'] # 日期特征
cat_features = ['kilometer',  'bodyType', 'brand', 'fuelType', 'gearbox', 'model', 'notRepairedDamage',  'regionCode'] # 类别特征
#################attention power add
######

num_features = ['v_' + str(i) for i in range(15)]  # power bin instead
#data[cat_features].astype(object)
#print(cat_features)
data.shape, data.columns
data["regionCode"] //= 100
data["power"] = np.log(data["power"]  + 1)
bin = [i/10  for i in range(-1, 71)]
data['power_bin'] = pd.cut(data['power'], bin, labels=False)
data['power_bin'][data['power'] >= 7] = 71
cat_features.append('power_bin')
data[['power_bin', 'power']].head()
data["notRepairedDamage"] = data["notRepairedDamage"].replace("-", "0.0").astype('float')
def date_proc(x):
    m = int(x[4:6])
    if m == 0:
        m = 1
    return x[:4] + '-' + str(m) + '-' + x[6:]


data['regDate'] = pd.to_datetime(data['regDate'].astype('str').apply(date_proc)) 
data['used_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') - 
                    pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days / 365
data['regDate' + '_year'] = data['regDate'].dt.year # 添加注册年份特征
cat_features.append('regDate_year')
data[[col for col in data.columns if col != 'price']].isnull().sum().sum()
data.info()
data = pd.get_dummies(data, columns=cat_features, dummy_na=True)
data.shape
data.isnull().sum().sum()
data.drop(columns=['price']).isnull().sum().sum()
data.columns
all_features = [col for col in data.columns if col not in date_features]
train_data = data[all_features][data['istrain'] == 1].drop(columns='istrain')
test_data = data[all_features][data['istrain'] == 0].drop(columns='istrain')
train_data.columns
n_train = train_data.shape[0]
train_features = torch.tensor(train_data.drop(columns=['SaleID','price']).values, dtype=torch.float)
test_features = torch.tensor(test_data.drop(columns=['SaleID','price']).values, dtype=torch.float)
train_labels = torch.tensor(train_data['price'].values, dtype=torch.float).view(-1, 1)
#loss = torch.nn.MSELoss()
loss = torch.nn.L1Loss()
def get_net(feature_num):
    
    net = nn.Sequential(
        nn.Linear(feature_num, 2048),
        nn.ReLU(),
        nn.Dropout(0.1),     
        nn.Linear(2048, 4096),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.2),  
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 128),
        nn.ReLU(),
    
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        )
    
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net
l1_loss = torch.nn.L1Loss()
def MAE(net, features, labels):
    with torch.no_grad():
        mae = l1_loss(net(features.to(device)), labels.to(device)).mean()
    return mae
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, device='cpu'):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    net = net.float()
    net = net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter: 
            X, y = X.to(device), y.to(device)
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(MAE(net, train_features, train_labels))
        if test_labels is not None:
            net.eval()
            test_ls.append(MAE(net, test_features, test_labels))
    return train_ls, test_ls
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    if k >=100:
        return X, y, X, y
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
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid
def k_fold(net, k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size, device='cpu', val_num=0):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(val_num, val_num + 1):
        data = get_k_fold_data(k, i, X_train, y_train)
        #net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size, device)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return net, train_l_sum / 1, valid_l_sum / 1
# k, num_epochs, lr, weight_decay, batch_size = 5, 5, 0.001, 0, 1000
# net, train_l, valid_l  = k_fold(net, k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
# print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))


for val_num in range(5):
    torch.cuda.empty_cache()
    best_valid_l = 10000
    best_model = None
    for num in range(10):
        print(num, 'loop')
        if best_model:
            net = best_model
        else:
            net = get_net(train_features.shape[1])
        k, num_epochs, lr, weight_decay, batch_size = 5, 1, 1e-4, 0, 256
        for i in range(25):
            if i<=20 and i % 5 == 4:
                lr *= 0.1
                batch_size *= 2
                net = best_model
            print(str(i) + 'th train on', device)
            net, train_l, valid_l  = k_fold(net, k, train_features, train_labels, 
                                            num_epochs, lr, weight_decay, batch_size, device, val_num
                                           )

            if valid_l < best_valid_l:
                best_model = net
                best_valid_l = valid_l 
                print("find best model, saving:")
                torch.save(net, str(val_num) + 'net.pkl')#整个网络
                #torch.save(net.state_dict(),'net_params.pkl')#网络的参数
                print('best model saved')
        print('train complete')
        
    net = best_model.to('cpu')
    preds = net(test_features).detach().numpy()
    temp = pd.DataFrame(preds.reshape(1, -1)[0], columns=['price'])
    sub = pd.DataFrame()
    sub['SaleID'] = test_data['SaleID']
    submission = sub.reset_index(drop = True)
    submission['price'] = temp
    submission.to_csv('/kaggle/working/' + str(val_num) + 'submission.csv', index=False)
    

# net = best_model.to('cpu')
# preds = net(test_features).detach().numpy()
# temp = pd.DataFrame(preds.reshape(1, -1)[0], columns=['price'])
# sub = pd.DataFrame()
# sub['SaleID'] = test_data['SaleID']
# submission = sub.reset_index(drop = True)
# submission['price'] = temp
# submission.to_csv('/kaggle/working/' + str(val_num) + 'submission.csv', index=False)
from IPython.display import FileLink
FileLink(str(4) + 'net.pkl')
pd.read_csv('4submission.csv').describe()
