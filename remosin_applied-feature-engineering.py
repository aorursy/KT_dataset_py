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
data_ibm = pd.read_csv('/kaggle/input/bot-detection/ibm_data.csv')
data_ibm.shape
data_ibm.head()
data_ibm['page_vw_ts'] = pd.to_datetime(data_ibm['page_vw_ts'])
data_ibm['page_vw_ts'].dt.year 
data_ibm.info()
#filtering and storing the date 
data_ibm['crm_dt'] = data_ibm['page_vw_ts'].dt.date
data_ibm.head()
#delete the page_vw_ts column
#data_ibm = data_ibm.drop('page_vw_ts', 1)
data_ibm.head()
import pandas as pd
import scipy as stats
import numpy as np
from statistics import mean 
import datetime
data_ibm.head(1)
data_ibm.dropna(subset=['city','st','operating_sys'], inplace=True)
data_ibm.device_type.fillna(value='unkown_device', inplace=True)
data_ibm.sec_lvl_domn.fillna(value='unkown_domain', inplace=True)
data_ibm.dropna(inplace=True)
data_ibm.drop(labels=['wk', 'mth', 'yr', 'crm_dt'], axis=1, inplace=True)
# Adding a Bounce_rate Column 
data_ibm["bounce_rate_%"] = ((data_ibm.VISIT - data_ibm.ENGD_VISIT)/data_ibm.VISIT)*100
data_ibm.head(2)
# Total of No. Of unique IP's
data_ibm.ip_addr.nunique()
# IP Addres's that has total views greater than 24 in a day
ip_views = pd.DataFrame(data_ibm.groupby('ip_addr').VIEWS.sum().sort_values())
unique_ip_address = list(ip_views[ip_views.VIEWS > 24].index)

# Limiting the Dataset to those rows that contain one of the ip's present in unique_ip_address 
#new_data_details = data_details[data_details.ip_addr.isin(unique_ip_address)]
new_data = data_ibm[data_ibm.ip_addr.isin(unique_ip_address)]

# Taking intersection of ip's
#unique_ip_address = list(new_data_details.ip_addr.unique())

# These are the filterd IP's on which we have to find Infomation.
print("No. Of unique ip's {}".format(len(unique_ip_address)))

# Examples of unique ip address
unique_ip_address[:10]
data_ibm.head(1)
new_data.head()
%%time
# This loop will find the views for each hour and also find the corresponding 
# bounce rate( i used median for bounce rate because mean was giving wrong results)
# and its infomation...

ip_hour_avg = pd.DataFrame([np.arange(24)])
ip_info = new_data.loc[new_data.ip_addr == unique_ip_address[0], 'ctry_name':'ip_addr'].head(1)
ip_hour_avg
count=0
bounce_rate = []
for ip_adres in (unique_ip_address):
    
    print("Iteration No. {}".format(count))
    time_data      = new_data[new_data.ip_addr==ip_adres]
    bounce_data    = new_data[new_data.ip_addr==ip_adres]
    unique_ip_info = new_data.loc[new_data.ip_addr == ip_adres, 'ctry_name':'ip_addr'].head(1)
    
    ip_info   = pd.concat([ip_info, unique_ip_info])
    b_rate    = bounce_data['bounce_rate_%'].median()
    bounce_rate.append(b_rate)
    
    hour_list = []
    for i in range(24):
        x = time_data[time_data.page_vw_ts.dt.hour == i].VIEWS.sum()
        hour_list.append(x)
    
    ip_dataframe  = pd.DataFrame([hour_list])    
    ip_hour_avg   = pd.concat([ip_hour_avg,ip_dataframe])
    count+=1
unique_ip_info.to_csv('unique_ip_info.csv')
ip_hour_avg.to_csv('ip_hour_avg.csv')
# Deleting first row from both DATASETS
ip_hour_avg = ip_hour_avg.iloc[1:,:]
ip_info     = ip_info.iloc[1:,:]
# Resetting Index 
ip_hour_avg = ip_hour_avg.set_index(np.arange(0,ip_hour_avg.shape[0]))
ip_info     = ip_info.set_index(np.arange(0,ip_info.shape[0])) 
# Finding Hourly_avg, daily_avg and Attaching corresponding Bounce_rate
ip_sum  = ip_hour_avg.sum(axis=1) #daily avg
ip_mean = ip_hour_avg.mean(axis=1) #hourly avg
ip_hour_avg['hour_avg']   = ip_mean
ip_hour_avg['daily_avg']  = ip_sum
ip_hour_avg['daily_avg']
ip_hour_avg['avg_bounce_rate'] = bounce_rate
#ip_hour_avg['avg_bounce_rate']
# Joining Both Datasets
new_ip_data = pd.merge(ip_info, ip_hour_avg, on=ip_info.index, how='outer')
new_ip_data.head()
new_ip_data.drop(['key_0'], axis=1, inplace=True)


# Changing Columns Name  to strings just for simplicity in future filtering... 
new_ip_data.columns = [                  'ctry_name',          'intgrtd_mngmt_name',
       'intgrtd_operating_team_name',                        'city',
                                'st',                'sec_lvl_domn',
                       'device_type',               'operating_sys',
                           'ip_addr',                             '0_hour',
                                   '1_hour',                             '2_hour',
                                   '3_hour',                             '4_hour',
                                   '5_hour',                             '6_hour',
                                   '7_hour',                             '8_hour',
                                   '9_hour',                            '10_hour',
                                  '11_hour',                            '12_hour',
                                  '13_hour',                            '14_hour',
                                  '15_hour',                            '16_hour',
                                  '17_hour',                            '18_hour',
                                  '19_hour',                            '20_hour',
                                  '21_hour',                            '22_hour',
                                  '23_hour',                    'hour_avg',
                         'daily_avg',             'avg_bounce_rate']

new_ip_data['date'] = (new_data.page_vw_ts.dt.date.value_counts().keys()[0]) 


# This loop will give a DATAFRAME that has all the information of each ip_address...
new_ip_data
new_ip_data.info()
# Taking the example Dataset
example_dataset = pd.DataFrame([data_ibm.iloc[0,:13]])
example_dataset.columns
example_dataset.drop(['Unnamed: 0'],1,inplace=True)
example_dataset.head()
example_dataset.columns = columns=[                  'ctry_name',          'intgrtd_mngmt_name',
       'intgrtd_operating_team_name',                        'city',
                                'st',                'sec_lvl_domn',
                       'device_type',               'operating_sys',
                           'ip_addr',                             'daily_info',
                                   'weekday_info',                             'date']
example_dataset
example_dataset.weekday_info = example_dataset.weekday_info.astype('object')
example_dataset.date = example_dataset.date.astype('object')
example_dataset.daily_info = example_dataset.daily_info.astype('object')
example_dataset.at[0,'daily_info'] = []
example_dataset.at[0,'weekday_info'] = []
example_dataset.at[0,'date'] = []
example_dataset
gl_data = example_dataset.copy()
gl_data.head()
# This loop will run in two parts first it check that the ip has visited us before or not based on that
# it will apppend its values for old IP's or will attach the new row for new ip's.

count=0
#gl_data = example_dataset.copy()
dt = str(new_ip_data.at[0,'date'])
wk_day = datetime.datetime.strptime(dt, '%Y-%m-%d').weekday()
new_ip_data.columns
len(unique_ip_address)
for ip_adrr in unique_ip_address:
    print("Iteration No. {}".format(count))
    
    if(ip_adrr in gl_data.ip_addr.unique()):
        
        idn = new_ip_data[new_ip_data.ip_addr == ip_adrr].index[0]
        vw =   new_ip_data.at[idn,'daily_avg']
        
        gl_data.at[idn,'daily_info'].append(vw)
        gl_data.at[idn,'weekday_info'].append(wk_day)
        #gl_data.at[idn,'date_info'].append(dt)
        gl_data.at[idn,'date'].append(dt)
        
    else:
        idx = new_ip_data[new_ip_data.ip_addr == ip_adrr].index[0]
        
        view = new_ip_data.at[idx,'daily_avg']
        
        a_data = new_data.loc[new_data.ip_addr == ip_adrr, 'ctry_name':'ip_addr'].head(1)
       # b_data = (pd.DataFrame([(ip_adrr,[view],[wk_day],[dt])],columns=['ip_addr',"daily_info",'weekday_info','date_info']))
        b_data = (pd.DataFrame([(ip_adrr,[view],[wk_day],[dt])],columns=['ip_addr',"daily_info",'weekday_info','date']))
        ip_info =pd.merge(a_data, b_data, on='ip_addr')
        
        gl_data = pd.concat([gl_data, ip_info], ignore_index=True)
        
    count+=1

    
gl_data
gl_data.to_csv("global_dataset.csv")
#gl_data.head()
gl_data.head()
gl_data.info()
# This function will give a dataframe that has all previous info. of single IP address
# INUPT: ip in string, new_ip_data, global_dataset
def ip_info(ip_address, global_data):
    idn = global_data[global_data.ip_addr == ip_address].index[0]
    d_vw =  pd.DataFrame(global_data.at[idn,'daily_info'])
    w_vw =  pd.DataFrame(global_data.at[idn,'weekday_info']) #take number not name
    d_in =  pd.DataFrame(global_data.at[idn,"date"])
    return pd.concat([d_vw,pd.concat([w_vw,d_in],axis=1)],axis=1)
#
def hour_rule(daily_dataset):
    ip_min = daily_dataset.iloc[:,9:33].min(axis=1)
    ip_max = daily_dataset.iloc[:,9:33].max(axis=1)
    ip_mean = daily_dataset.hour_avg
    daily_dataset["hour_per"] = pd.DataFrame([round((abs(ip_min - ip_mean )/ip_mean), ndigits=2),round((abs(ip_max - ip_mean )/ip_mean), ndigits=2)]).max()
    #daily_dataset.insert(daily_dataset.shape[1], "bot_prob", 0.00)
    daily_dataset['bot_prob']=0.00
    daily_dataset.loc[daily_dataset.hour_per < 10.00, "bot_prob"] += 0.15
    daily_dataset.drop(['hour_per'], axis=1, inplace=True)

def max_absolute(global_dataset,idx_g):
    return max(round((abs(min(global_dataset.at[idx_g,"daily_info"]) - mean(global_dataset.at[idx_g,"daily_info"]) )/mean(global_dataset.at[idx_g,"daily_info"])), ndigits=2),round((abs(max(global_dataset.at[idx_g,"daily_info"]) - mean(global_dataset.at[idx_g,"daily_info"]) )/mean(global_dataset.at[idx_g,"daily_info"])), ndigits=2))
     
def wk_rule(ip,  global_dataset):
    x = ip_info(ip, global_dataset)
    columns=['daily_info','weekday_info','date']
    x.columns=columns
    a = x[x.weekday_info < 5].weekday_info.mean()
    b = x[x.weekday_info > 4].weekday_info.mean()
    return abs(a-b)/max(a,b) 
# Rule_1 Labeling on basis of hour
hour_rule(new_ip_data)

count=0
for ip in unique_ip_address:
    print("Iteration No. {}".format(count))
    
    if(ip in gl_data.ip_addr.unique()):
        
        idx_n = new_ip_data[new_ip_data.ip_addr == ip].index[0]  #Index of ip in daily_dataset
       
        #Rule_2 Labeling on basis of daily_avg
        idx_g = gl_data[gl_data.ip_addr == ip].index[0]  #Index of ip in global_dataset
        if (max_absolute(gl_data, idx_g) < 10.00):
            new_ip_data.at[idx_n,"bot_prob"] += 0.15
            
        #Rule_3 Labeling on basis of weekday_avg
        if (wk_rule(ip,gl_data) < 10.00):
            new_ip_data.at[idx_n,"bot_prob"] += 0.15
            
        #Rule_4 Labeling on basis of bounce rate
        if(new_ip_data.at[idx_n,"avg_bounce_rate"] >= 80.0):
            new_ip_data.at[idx_n,"bot_prob"] += 0.10
            
        #Rule_5 Labeling on basis of Operating system
        if(new_ip_data.at[idx_n,"operating_sys"] == 'LINUX'):
            new_ip_data.at[idx_n,"bot_prob"] += 0.10
            
        
    else:
        # If we dont have previous information of an ip_address then we are left with only three rules..
        # a. Hourly_average
        # b. bounce_rate
        # c. Operating_system
        
        idx_n = new_ip_data[new_ip_data.ip_addr == ip].index[0]  #Index of ip in daily_dataset
        
        # Labeling on basis of bounce rate
        if(new_ip_data.at[idx_n,"avg_bounce_rate"] >= 80.0):
            new_ip_data.at[idx_n,"bot_prob"] += 0.10
        
        # Labeling on basis of Operating system
        if(new_ip_data.at[idx_n,"operating_sys"] == 'LINUX'):
            new_ip_data.at[idx_n,"bot_prob"] += 0.10
        
        
    count+=1
    
    
#This is the our final Dataset on which we train the model
new_ip_data.sample(10)
new_ip_data[new_ip_data.bot_prob > 0.1]
new_ip_data.to_csv("botprob.csv")
new_ip_data['bot_prob'].value_counts()
pd.pandas.set_option("display.max_columns",None)
botprob=pd.read_csv("botprob.csv")
botprob
botprob.info()
cat_features=[feature for feature in botprob.columns if botprob[feature].dtype =='O']
cat_features
cat_features.remove('date')
cat_features.remove('ip_addr')
cat_features
for feature in cat_features:
    freq=botprob[feature].value_counts()
    print(freq)
    botprob[feature].replace(freq,inplace=True)
botprob
botprob.drop('Unnamed: 0',1,inplace=True)
botprob.head()
botprob['ip_addr']=botprob['ip_addr'].astype('category')
botprob['ip_addr']=botprob['ip_addr'].cat.codes
botprob.head()
botprob.drop(['date'],1,inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
botprob['target']=0
for i in range(len(botprob)):
    if botprob['bot_prob'][i]<=0.25:
        botprob['target'][i]=0
    else:
        botprob['target'][i]=1
botprob.head()
botprob.shape
sns.heatmap(botprob.corr())
botprob.drop('intgrtd_mngmt_name',axis=1,inplace=True)
botprob.isnull().any()
