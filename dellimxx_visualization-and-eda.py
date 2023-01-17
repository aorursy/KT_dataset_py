import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('../input/Data/Stocks')
files= [x for x in os.listdir() if x.endswith('.txt') and os.path.getsize(x) > 600000]
files
mkr = pd.read_csv(files[0],sep=',',index_col='Date')
ibm = pd.read_csv(files[1],sep=',',index_col='Date')
xom = pd.read_csv(files[2],sep=',',index_col='Date')
ba = pd.read_csv(files[3],sep=',',index_col='Date')
dis = pd.read_csv(files[4],sep=',',index_col='Date')
mcd = pd.read_csv(files[5],sep=',',index_col='Date')
utx = pd.read_csv(files[6],sep=',',index_col='Date')
ge = pd.read_csv(files[7],sep=',',index_col='Date')
jnj = pd.read_csv(files[8],sep=',',index_col='Date')
hpq = pd.read_csv(files[9],sep=',',index_col='Date')
ko = pd.read_csv(files[10],sep=',',index_col='Date')
mo = pd.read_csv(files[11],sep=',',index_col='Date')
mkr.head(10)
# create close DataFrame
close_price = pd.DataFrame()
close_price['mkr'] = mkr['Close']
close_price['ibm'] = ibm['Close']
close_price['xom'] = xom['Close']
close_price['ba'] = ba['Close']
close_price['dis'] = dis['Close']
close_price['mcd'] = mcd['Close']
close_price['utx'] = utx['Close']
close_price['ge'] = ge['Close']
close_price['jnj'] = jnj['Close']
close_price['ko'] = ko['Close']
close_price['mo'] = mo['Close']
close_price = close_price.fillna(method='ffill')
close_price.index =close_price.index.astype('datetime64[ns]')
close_price.describe()
close_price.head(10)
%matplotlib inline
_ = pd.concat([close_price['mkr'],close_price['ibm'],close_price['xom'],close_price['ba'],close_price['dis'],close_price['mcd'],close_price['utx'],close_price['ge'],close_price['jnj'],close_price['ko'],close_price['mo']],axis=1).plot(figsize=(20,15),grid=True)
# Remove 1994
from datetime import datetime
close_price = close_price[close_price.index.year>=1994]
%matplotlib inline
_ = pd.concat([close_price['mkr'],close_price['ibm'],close_price['xom'],close_price['ba'],close_price['dis'],close_price['mcd'],close_price['utx'],close_price['ge'],close_price['jnj'],close_price['ko'],close_price['mo']],axis=1).plot(figsize=(20,15),grid=True)
company_list = close_price.columns
for idx in company_list:
    close_price[idx+'_scale'] = close_price[idx]/max(close_price[idx])
_ = pd.concat([close_price['mkr_scale'],close_price['ibm_scale'],close_price['xom_scale'],close_price['ba_scale'],close_price['dis_scale'],close_price['mcd_scale'],close_price['utx_scale'],close_price['ge_scale'],close_price['jnj_scale'],close_price['ko_scale'],close_price['mo_scale']],axis=1).plot(figsize=(20,15),grid=True)
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix

fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(15)

for i in company_list:
    if '_scale' not in i:
        _ = autocorrelation_plot(close_price[i],label=i)
_ = scatter_matrix(pd.concat([close_price['mkr_scale'],close_price['ibm_scale'],close_price['xom_scale'],close_price['ba_scale'],close_price['dis_scale'],close_price['mcd_scale'],close_price['utx_scale'],close_price['ge_scale'],close_price['jnj_scale'],close_price['ko_scale'],close_price['mo_scale']],axis=1),figsize=(20,15),diagonal='kde')
log_data = pd.DataFrame()
for idx in company_list:
    if '_scale' not in idx:
        log_data[idx+'_log'] = np.log(close_price[idx]/close_price[idx].shift())
log_data.describe()
_ = pd.concat([log_data['mkr_log'],log_data['ibm_log'],log_data['xom_log'],log_data['ba_log'],log_data['dis_log'],log_data['mcd_log'],log_data['utx_log'],log_data['ge_log'],log_data['jnj_log'],log_data['ko_log'],log_data['mo_log']],axis=1).plot(figsize=(20,15),grid=True)
fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(15)

list_log = log_data.columns

for i in list_log:
    _ = autocorrelation_plot(log_data[i])
#Let's take a look at the basic correlation.
log_data.corr()
yesterday = pd.DataFrame()
yesterday['mkr'] = log_data['mkr_log'].shift(1)
yesterday['ibm'] = log_data['ibm_log'].shift(1)
yesterday['xom'] = log_data['xom_log'].shift(1)
yesterday['ba'] = log_data['ba_log'].shift(1)
yesterday['dis'] = log_data['dis_log'].shift(1)
yesterday['mcd'] = log_data['mcd_log'].shift(1)
yesterday['utx'] = log_data['utx_log'].shift(1)
yesterday['ge'] = log_data['ge_log'].shift(1)
yesterday['jnj'] = log_data['jnj_log'].shift(1)
yesterday['ko'] = log_data['ko_log'].shift(1)
yesterday['mo'] = log_data['mo_log'].shift(1)
all = pd.concat([log_data,yesterday],axis=1)
all.corr()
