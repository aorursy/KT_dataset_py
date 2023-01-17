# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 读取测试文件
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
# display(train.head(100))
# display(train.describe())
print("OK")
# 显示国家名字
print(train.drop_duplicates('Country_Region')['Country_Region'].head(-10))
# choose one country to analysis  
# 选取一个国家，按照日期读入并且绘图。 左边 感染人数，右边 死亡人数
Country_name = 'Turkey'
confirm_date_coutry = train[train['Country_Region']==Country_name].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_coutry = train[train['Country_Region']==Country_name].groupby(['Date']).agg({'Fatalities':['sum']})
# confirm_date_coutry = train[:].groupby(['Date']).agg({'ConfirmedCases':['sum']})
# fatalities_total_date_coutry = train[:].groupby(['Date']).agg({'Fatalities':['sum']})
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))
confirm_date_coutry.plot(ax=ax1)
ax1.set_title("%s confirmed cases" %Country_name, size=13)
fatalities_total_date_coutry.plot(ax=ax2)
# ＲＮＮ　模型　
import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size = 1,
            hidden_size = 2,
            num_layers = 1,
            nonlinearity='relu',
        )
        self.out = nn.Linear(2,1)
    def forward(self,x,h):
        out,h = self.rnn(x,h)
        pre = self.out(out)
        return pre,h
rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)
loss_func = nn.MSELoss()
h_state = None
# 读取数据进入。Ｘ_train 训练的输入数据，y_train 训练的输出数据。　seq 该国家的日期总天数。
print(type(confirm_date_coutry))
z = np.array(confirm_date_coutry['ConfirmedCases'])
x = torch.tensor(z)
seq = x.size()[0] - 10


X_trains = x[0:seq]
y_trains = x[10:seq + 10]

print("MAX = %d" %(max(x)))
print("seq = %d" %seq)
# 训练模型， 每次读入10天进行训练
import random
loss_save = []
for epoch in range(100):
    if(epoch % 100 == 99):
        print(epoch+1)
    for i in range(5):
        # step = random.randint(0,seq - 1)
        step= i
        X_train = X_trains[10 * step : 10 * (step + 1),np.newaxis]
        y_train = y_trains[10 * step : 10 * (step + 1),np.newaxis]

        X_train = torch.tensor(X_train,dtype=torch.float32)
        y_train = torch.tensor(y_train,dtype=torch.float32)
        pre,h_state = rnn(X_train,h_state)
        h_state = h_state.detach()
        loss = loss_func(pre, y_train)
        loss_save.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         print(X_train)
#         print(y_train)
#         print("step = "f'{step}')
print("end")
#显示训练数据的格式
X_train = X_trains[10 * step : 10 * (step + 1),np.newaxis,]
print(X_train.size())
# print(y_train)
# 查看 loss的变化，判断有没有收敛
plt.plot(loss_save)
#通过训练数据去进行预测 预测过程中是 每次输入一天， 然后输出下一天的预测结果
pre_date = x.size()[0]
confirm_date_coutry = train[train['Country_Region']==Country_name].groupby(['Date']).agg({'ConfirmedCases':['sum']})
one_coutry = torch.tensor(np.array(confirm_date_coutry)) /MAX
pred= []
true = []
for i in range(pre_date):
#     b,h = rnn(torch.tensor(one_coutry[i,np.newaxis,np.newaxis],dtype=torch.float32),h_state)
#     true.append(one_coutry[i,0]*MAX)
#     pred.append(b[0,0,0]*MAX)
    b,h = rnn(torch.tensor(x[i,np.newaxis,np.newaxis],dtype=torch.float32),h_state)
#     true.append(x[i,0]*MAX)
#     pred.append(b[0,0,0]*MAX)
    true.append(x[i,0])
    pred.append(b[0,0,0])
plt.subplot(1,2,1)
plt.plot(pred,'bo')
plt.subplot(1,2,2)
plt.plot(true,'bo')
# 通过预测模型， 只给一天的输入，让他递推的输出连续几天的感染人数
ans_save = []
# ans,h = rnn(torch.tensor([[[1.0]]]),h_state)
# print(ans)
# print(MAX)
print("x[-10] = " f'{x[-10]}')
print("x[-9] = " f'{x[-9]}')
ans,h = rnn(torch.tensor(x[-10,np.newaxis,np.newaxis],dtype=torch.float32),h_state)
for i in range(10):
    # ans = torch.tensor([[[i/10]]])
    # print(ans)
    ans,h = rnn(ans,h_state)
    # print(ans)
    ans_save.append(ans[0,0,0])
plt.plot(ans_save,'bo')
# ans = torch.tensor([[[0.008]]])
# ans,h = rnn(ans,h_state)
# print(ans)
# 查看预测模型和实际模型的差别
plt.plot([true[i]- pred[i] for i in range(len(true))],'bo')
