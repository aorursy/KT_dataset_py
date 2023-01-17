# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
aapl = pd.read_csv('../input/stocknew/AAPL_CLOSE',index_col='Date',parse_dates=True)
cisco = pd.read_csv('../input/stocknew/GOOG_CLOSE',index_col='Date',parse_dates=True)
ibm = pd.read_csv('../input/stocknew/IBM_CLOSE',index_col='Date',parse_dates=True)
amzn = pd.read_csv('../input/stocknew/AMZN_CLOSE',index_col='Date',parse_dates=True)
stocks = pd.concat([aapl,cisco,ibm,amzn],axis=1)
stocks.columns = ['aapl','cisco','ibm','amzn']
log_ret=np.log (stocks/stocks.shift(1))
log_ret.head()
log_ret.plot()
plt.figure(figsize=(15,8))
ax=sns.heatmap(stocks.corr(),annot = True,)
np.random.seed(101)

num_ports=15000
all_weight=np.zeros((num_ports,len(stocks.columns)))
ret_arr=np.zeros(num_ports)
vol_arr=np.zeros(num_ports)
sharpe_arr=np.zeros(num_ports)
 
for ind in range(num_ports):
    weights=np.array(np.random.random(4))
    weights=weights/np.sum(weights)
    all_weight[ind,:]=weights
    ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)
    vol_arr[ind]=np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
    sharpe_arr[ind]=ret_arr[ind]/vol_arr[ind]
 
m=sharpe_arr.argmax() 
sharpe_arr.max()
ret_arr[m]
vol_arr[m]   
all_weight[m,:]
max_sn_ret=ret_arr[m]
max_sn_vol=vol_arr[m]   
plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label="sharpe Ratio")
plt.xlabel("volatility")
plt.ylabel("Return")
plt.scatter(max_sn_vol,max_sn_ret,c="red",s=50,edgecolors="black")