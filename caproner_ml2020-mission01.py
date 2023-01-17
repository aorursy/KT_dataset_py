# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/ml2020spring-hw1/train.csv', encoding='big5')
train_data.info()
train_data.head(20)
train_data.describe()
train_data = train_data.drop(['測站'], axis=1).copy()
train_data.head(20)
test_data = pd.read_csv('/kaggle/input/ml2020spring-hw1/test.csv', 
                        encoding='big5', names = ['id', '測項', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
test_data.info()
test_data.head(20)
train_s = train_data[['日期', '測項', '8']].copy()
train_s['日期'] = pd.to_datetime(train_s['日期'] + ' ' + '8' +':00:00')
train_s = train_s.pivot(index='日期', columns='測項', values='8')
train_s
train_data_fixed = pd.DataFrame()
for i in range(24):
    train_data_slice = train_data[['日期', '測項', str(i)]].copy()
    train_data_slice['日期'] = pd.to_datetime(train_data_slice['日期'] + ' ' + str(i) +':00:00')
    train_data_slice = train_data_slice.pivot(index='日期', columns='測項', values=str(i))
    train_data_fixed = pd.concat([train_data_fixed, train_data_slice])

train_data_fixed
train_data_fixed = train_data_fixed.replace('NR', '0').astype('float64').sort_index().reset_index().drop(['日期'], axis=1)
data_mean = train_data_fixed.mean().copy()
data_std = train_data_fixed.std().copy()
for i in range(18):
    train_data_fixed.iloc[:,[i]] = (train_data_fixed.iloc[:,[i]] - data_mean[i]) / data_std[i]
train_data_fixed.describe()
txx = train_data_fixed

tx = train_data_fixed.copy()
tx.columns = tx.columns + '_0'
for i in range(1,10):
    ty = train_data_fixed.copy()
    if i == 9:
        ty = ty[['PM2.5']]
        # 结果列不需要标准化，需要放大回去
        ty = ty * data_std['PM2.5'] + data_mean['PM2.5']
    ty.columns = ty.columns + '_' + str(i)
    for j in range(i):
        ty = ty.drop([j])
    tx = pd.concat([tx, ty.reset_index().drop(['index'], axis=1)], axis=1)

for i in range(12):
    for j in range(9):
        tx = tx.drop([480*(i+1)-9+j])
train_data = tx
train_data
train_data.describe()
train_data_name = train_data.columns.copy()
for i in train_data_name:
    if i == 'PM2.5_9':
        continue
    train_data[i + '_*2'] = (train_data[i] ** 2)
    train_data[i + '_*3'] = (train_data[i] ** 3) / 10
train_data
train_x = train_data.drop(['PM2.5_9'], axis=1)
train_y = train_data[['PM2.5_9']]
x = np.hstack((train_x.values, np.ones((np.size(train_x.values,0), 1), 'double')))
y = train_y.values
print(np.size(x,0), np.size(x,1))
print(np.size(y,0), np.size(y,1))
def get_loss(_x, _y, _theta):
    return np.sum((_y-_x.dot(_theta))**2)

theta = np.random.random((np.size(x,1), 1))
learning_rate = 41e-9
regular_param = 1

train_X = x[:4239]
train_Y = y[:4239]
vari_X = x[4239:]
vari_Y = y[4239:]

x_mix = train_X.T.dot(train_X)
x_sub = train_X.T.dot(train_Y)

def get_gradient(_x, _y, _theta):
    return x_mix.dot(_theta)-x_sub + (regular_param * _theta)

for i in range(1000001):
    theta = theta - learning_rate*get_gradient(train_X, train_Y, theta)
    if i % 20000 == 0:
        print(i, get_loss(train_X, train_Y, theta) / np.size(train_Y,0), get_loss(vari_X, vari_Y, theta) / np.size(vari_Y,0))
test_data_id = test_data['id']
test_data['id'] = test_data['id'].str.split('_', expand = True)[1].astype('int')
test_data_fixed = pd.DataFrame()
for i in range(9):
    test_data_slice = test_data[['id', '測項', str(i)]].copy()
    test_data_slice = test_data_slice.pivot(index='id', columns='測項', values=str(i))
    test_data_slice.columns = test_data_slice.columns + '_' + str(i)
    for j in range(18):
        test_data_slice.iloc[:,[j]] = (test_data_slice.iloc[:,[j]].replace('NR', '0').astype('float64') - data_mean[j]) / data_std[j]
    test_data_fixed = pd.concat([test_data_fixed, test_data_slice], axis=1)

test_data_fixed = test_data_fixed.replace('NR', '0').astype('float64').reset_index().drop(['id'], axis=1)
test_data_fixed
test_data_name = test_data_fixed.columns.copy()
for i in test_data_name:
    if i == 'PM2.5_9':
        continue
    test_data_fixed[i + '_*2'] = test_data_fixed[i] ** 2
    test_data_fixed[i + '_*3'] = (test_data_fixed[i] ** 3) / 10
test_data_fixed
test_x = np.hstack((test_data_fixed.values, np.ones((np.size(test_data_fixed.values,0), 1), 'double')))
print(np.size(test_x,0), np.size(test_x,1))
test_y = test_x.dot(theta)
test_y
submission = pd.DataFrame({
        "id": test_data_id.unique(),
        "value": test_y.T[0]
    })
submission.to_csv('/kaggle/working/submission.csv', index=False)
theta