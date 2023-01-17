# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#### 先来看看数据是什么样的
train_data = pd.read_csv('/kaggle/input/ml2020spring-hw1/train.csv',encoding = 'big5')
test_data = pd.read_csv('/kaggle/input/ml2020spring-hw1/test.csv',encoding = 'big5',names = ['id', '測項', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
test_data['id'] = test_data['id'].str.split('_',expand = True)[1].astype('int')
#train_data.info()
#train_data.head(20)
test_data
#先看看CH4
ch4 = train_data[train_data['測項'].isin(['CH4'])]
ch4.head()
#看看RAINFALL
rainfall = train_data[train_data['測項'].isin(['RAINFALL'])]
#rainfall.info()
#把RAINFALL按照上面说的那样replace一下
train_data.replace("NR",0, inplace = True)
test_data.replace("NR",0, inplace = True)
train_data.head(18)
rainfall = train_data[train_data['測項'].isin(['RAINFALL'])]
#rainfall.info()
#
for num in range(0,23) :
    num = str(num)
    train_data[num] = train_data[num].astype(float)

train_data.head()
#先置换一下行和列
#train_data_reshaped = train_data.pivot_table(train_data,index=[u'0'],columns="測項")
#train_data_reshaped.head()
#columns=('idx','degree','weight','diameter')
train_data_re = pd.DataFrame()
for num in range(24):
    num = str(num)
    train_data_re = train_data_re.append(
                pd.DataFrame(train_data[num].values.reshape(1,-1), 
                index=['id_'+num], 
                columns=train_data['測項'].values)
                )
train_data_re.head(25)

#把无用数据删掉
train_data.head()
train_data_v2 = train_data.copy()
#删掉了测站
train_data_v2 = train_data_v2.drop(columns = '測站')
train_data_v2

#train_data进行转换
train_data_v2

train_data_new = pd.DataFrame()
for x in range(24):
    train_data_first = train_data_v2[['日期','測項',str(x)]].copy()
    train_data_first['日期'] = pd.to_datetime(train_data_first['日期']+' '+str(x)+':00:00')
    train_data_first = train_data_first.pivot(index = '日期',columns = '測項', values = str(x))
    train_data_new = pd.concat([train_data_new,train_data_first])
train_data_new = train_data_new.astype('float64').sort_index().reset_index().drop(['日期'], axis = 1)
train_data_new
#feature scaling for train_data
#(X-mean)/std
train_mean = train_data_new.mean().copy()
train_std = train_data_new.std().copy()
train_data_new1 = train_data_new.copy()
for liecolumn in train_data_new:
        train_data_new[liecolumn] = (train_data_new[liecolumn] - train_mean[liecolumn])/train_std[liecolumn]
        #print(liecolumn,train_data_new[liecolumn])
        
train_data_new
tx = train_data_new.copy()
tx.columns = tx.columns + '_0'
for i in range(1,10):
    ty = train_data_new.copy()
    if i == 9:
        ty = ty[['PM2.5']]
        # 结果列不需要标准化，需要放大回去
        ty = ty * train_std['PM2.5'] + train_mean['PM2.5']
    ty.columns = ty.columns + '_' + str(i)
    for j in range(i):
        ty = ty.drop([j])
    tx = pd.concat([tx, ty.reset_index().drop(['index'], axis=1)], axis=1)

for i in range(12):
    for j in range(9):
        tx = tx.drop([480*(i+1)-9+j])
train_data = tx
train_data.describe()
test_data_new = pd.DataFrame()
for i in range(9):
    test_data_slice = test_data[['id', '測項', str(i)]].copy()
    test_data_slice = test_data_slice.pivot(index='id', columns='測項', values=str(i))
    test_data_slice.columns = test_data_slice.columns + '_' + str(i)
    for j in range(18):
        test_data_slice.iloc[:,[j]] = (test_data_slice.iloc[:,[j]].replace('NR', '0').astype('float64') - train_mean[j]) / train_std[j]
    test_data_new = pd.concat([test_data_new, test_data_slice], axis=1)

test_data_new = test_data_new.replace('NR', '0').astype('float64').reset_index().drop(['id'], axis=1)
test_data_new
#在后面都要加上一个常数 方便后面做gradient descent的时候使用
train_x = train_data.drop(['PM2.5_9'],axis = 1)
train_y = train_data[['PM2.5_9']]
x = np.hstack((train_x.values,np.ones((np.size(train_x.values,0),1),'double')))
y = train_y.values
x
y
# 预测方法
# LOSS function
def loss(x,y,data):
    return np.sum((y- x @ data)**2)

# 梯度下降
def gradientDescent(x,y,data):
    return ((train_x_x) @ data) - (train_x_y) +(1 * data)
data = np.random.random((np.size(x,1),1))
data
#学习速率
learning_rate = 0.00000006
regular_one = 1

#把训练数据分成四份 按照3:1的比例设置为 训练集和验证集
train_X = x[:4320]
train_Y = y[:4320]
vari_X = x[4320:]
vari_Y = y[4320:]

#GD =  gradientDescent(train_X,train_Y,data)
#把gradientdescent里的参数提取出来 这样就不用在迭代过程中反复计算 可以节约时间
train_x_x = train_X.T @ train_X
train_x_y = train_X.T @ train_Y

for i in range(2000001):
    data = data - learning_rate * gradientDescent(train_X,train_Y,data)
    if i %50000 == 0 :
        #输出训练集误差和验证集误差 
        print(i,loss(train_X,train_Y,data) / np.size(train_Y,0) , loss(vari_X,vari_Y,data) / np.size(train_Y,0))


print(np.size(data,0), np.size(data,1))
test_x = np.hstack((test_data_new.values, np.ones((np.size(test_data_new.values,0), 1), 'double')))
print(np.size(test_x,0), np.size(test_x,1))

test_y = test_x @ (data)
test_y


test_data_id = test_data['id']

submission = pd.DataFrame({
        "id": test_data_id.unique(),
        "value": test_y.T[0]
    })
submission.to_csv('/kaggle/working/submission.csv', index=False)