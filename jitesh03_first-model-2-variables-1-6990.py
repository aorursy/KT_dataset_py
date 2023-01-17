import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize
import datetime
import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import calendar
import time
import os
import gc
gc.enable()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
def load_df(csv_path = '../input/train.csv' , nrows = None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

def int2str(x):
    return '{}'.format(x)

def str_to_date(x ,baseformat = "%Y%m%d"):
    return datetime.datetime.strptime(x , baseformat).date()

def constant_cols(data):
    constant_columns = [x for x in data.columns if data[x].nunique() <= 1]
    return constant_columns

def check_missing_vals(data , t = 0.5):
    missing_vals = [c for c in data.columns if data[c].isnull().sum()/data.shape[0] >= t and c != 'totals.transactionRevenue']
    return missing_vals

def conv2lower(x):
    return str(x).lower()

def to_lower(data):
    for c in data.columns:
        if data[c].dtype == 'object':
            data[c] = data[c].apply(conv2lower)
    return data

def rep_text(data , txt):
    try:
        data = data.replace(txt , "NA")
    except:
        pass
    return data

def find_dow(x):
    return calendar.day_name[x.weekday()]

def find_month(x):
    return x.month

def replace_missing_vals(data):
    exclude_list = ['date','fullVisitorId','sessionId','visitId']
    for c in data.columns:
        if data[c].dtype == 'object' and c not in exclude_list:
            data[c] = data[c].fillna('NA')
        elif data[c].dtype != 'object' and c not in exclude_list:
            data[c]=data[c].fillna(0)
    return data

def filter_cols(data):
    exclude_list = ['date','fullVisitorId','sessionId','visitId']
    cat_list = []
    num_list = []
    for c in data.columns:
        if data[c].dtype =='object' and c not in exclude_list:
            cat_list.append(c)
        elif data[c].dtype != 'object' and c not in exclude_list:
            num_list.append(c)
    return cat_list , num_list
            
    
def cat2num(data):
    exclude_list = ['date','fullVisitorId','sessionId','visitId']
    le = LabelEncoder()
    for c in data.columns:
        if data[c].dtype =='object' and c not in exclude_list:
            data[c] = le.fit_transform(data[c])
    return data

def chk_num_lvls(data):
    lvls_dict = {}
    exclude_list = ['date','fullVisitorId','sessionId','visitId']
    for each in data.columns:
        if data[each].dtype == 'object' and each not in exclude_list:
            lvls_dict[each] = data[each].nunique()
    return lvls_dict

def find_req_levels(data):
    exclude_list = ['date','fullVisitorId','sessionId','visitId']
    target = 'totals.transactionRevenue'
    for each in data.columns:
        if data[each].dtype == 'object' and each not in exclude_list:
            grouped = data.groupby([each])[target].mean()
            grouped = grouped.reset_index()
            grouped.columns = ['field' , 'value']
            grouped = grouped[grouped['value'] != 0]
            useful_levels = grouped['field'].values
            #print(useful_levels , len(useful_levels))
            x1 = data[each].values
            #print(x1)
            x2 = []
            for x in x1:
                if x in useful_levels:
                    x2.append(x)
                else:
                    x2.append('others')
            #print(x2)
            data[each] = x2
    return data
%%time
df_train = load_df("../input/train.csv")
df_test = load_df("../input/test.csv")
print("Data Reading Completed!!")
df_combined = pd.concat([df_train,df_test])
train_length = len(df_train)
del df_train , df_test
%%time
df_combined["date"] = pd.to_datetime(df_combined["date"],format="%Y%m%d")
df_combined["visitStartTime"] = pd.to_datetime(df_combined["visitStartTime"],unit='s')
float_cols = ['totals.bounces','totals.hits','totals.newVisits','totals.pageviews','totals.transactionRevenue','totals.visits']
for each in float_cols:
    try:
        df_combined[each] = df_combined[each].astype('float')
    except:
        pass
df_combined.head(10) 
#Try and explore the dataset visually
#Analyzing transaction revenues

df_combined['totals.transactionRevenue']=df_combined['totals.transactionRevenue'].fillna(0)
df_combined['totals.transactionRevenue'] = df_combined['totals.transactionRevenue'].apply(lambda x: x/1000000)
# x = df_combined['totals.transactionRevenue'][df_combined['totals.transactionRevenue'] != 0]
# fig , (ax1,ax2) = plt.subplots(1,2, figsize = (15,5))
# ax1.hist(x , color = 'coral')
# ax1.set_title('Non-zero Revenue')

# ax2.hist(np.log1p(x) , color = 'coral')
# ax2.set_title('Natural Log of Non-zero Revenue')
#Plot of revenue and sessions by date
grouped_data = df_combined.groupby('date', as_index = False).agg({'sessionId':'count',
                                                                  'totals.bounces':'sum',
                                                                  'totals.hits':'sum',
                                                                  'totals.newVisits':'sum',
                                                                  'totals.pageviews':'sum',
                                                                  'totals.visits':'sum',
                                                                  'totals.transactionRevenue':'sum'})
grouped_data.columns = ['Date','no_of_sessions','bounces','hits','newvisits','pageviews','visits','revenue']
#grouped_data[grouped_data['revenue'] != 0]
fig , axes = plt.subplots(5,1,figsize=(30,30) , sharex = 'col')
axes[0].plot(grouped_data['Date'] , grouped_data['no_of_sessions'] , color = 'coral')
axes[0].plot(grouped_data['Date'] , grouped_data['revenue'] , color = 'black')
axes[0].set_ylabel('PageView/Revenue')

axes[1].plot(grouped_data['Date'] , grouped_data['bounces'] , color = 'coral')
axes[1].plot(grouped_data['Date'] , grouped_data['revenue'] , color = 'black')
axes[1].set_ylabel('Sessions/Revenue')

axes[2].plot(grouped_data['Date'] , grouped_data['newvisits'] , color = 'coral')
axes[2].plot(grouped_data['Date'] , grouped_data['revenue'] , color = 'black')
axes[2].set_ylabel('NewVisits/Revenue')

axes[3].plot(grouped_data['Date'] , grouped_data['pageviews'] , color = 'coral')
axes[3].plot(grouped_data['Date'] , grouped_data['revenue'] , color = 'black')
axes[3].set_ylabel('PageViews/Revenue')

axes[4].plot(grouped_data['Date'] , grouped_data['visits'] , color = 'coral')
axes[4].plot(grouped_data['Date'] , grouped_data['revenue'] , color = 'black')
axes[4].set_ylabel('Visits/Revenue')
fig , axes = plt.subplots(2,3 , figsize = (15,5) , sharey = 'col')
axes[0,0].scatter(grouped_data['no_of_sessions'] , grouped_data['revenue'] , color = 'coral')
axes[0,0].set_xlabel('Sessions')
axes[0,0].set_ylabel('Revenue')

axes[0,1].scatter(grouped_data['bounces'] , grouped_data['revenue'] , color = 'coral')
axes[0,1].set_xlabel('Bounces')
axes[0,1].set_ylabel('Revenue')

axes[0,2].scatter(grouped_data['newvisits'] , grouped_data['revenue'] , color = 'coral')
axes[0,2].set_xlabel('NewVisits')
axes[0,2].set_ylabel('Revenue')

axes[1,0].scatter(grouped_data['pageviews'] , grouped_data['revenue'] , color = 'coral')
axes[1,0].set_xlabel('PageViews')
axes[1,0].set_ylabel('Revenue')

axes[1,1].scatter(grouped_data['visits'] , grouped_data['revenue'] , color = 'coral')
axes[1,1].set_xlabel('Visits')
axes[1,1].set_ylabel('Revenue')

axes[1,2].scatter(grouped_data['hits'] , grouped_data['revenue'] , color = 'coral')
axes[1,2].set_xlabel('Hits')
axes[1,2].set_ylabel('Revenue')

#Plot Revenue by device categories
df_combined = rep_text(df_combined , "not available in demo dataset")
df_combined = find_req_levels(df_combined)
grouped_data_device = df_combined.groupby('device.browser', as_index = False)['totals.transactionRevenue'].sum()
grouped_data_device.columns = ['browser' , 'revenue']

grouped_data_os = df_combined.groupby('device.operatingSystem', as_index = False)['totals.transactionRevenue'].sum()
grouped_data_os.columns = ['os' , 'revenue']

fig , axes = plt.subplots(1,2 , figsize = (15,5))
axes[0].bar(grouped_data_device['browser'] , grouped_data_device['revenue'] , color = 'coral')
axes[0].set_xlabel('Browser')
axes[0].set_ylabel('Revenue')

axes[1].bar(grouped_data_os['os'] , grouped_data_os['revenue'] , color = 'coral')
axes[1].set_xlabel('OS')
axes[1].set_ylabel('Revenue')
geo_continent = df_combined.groupby('geoNetwork.continent', as_index = False)['totals.transactionRevenue'].sum()
geo_continent.columns = ['continent' , 'revenue']
geo_rev = geo_continent['revenue'].sum()
geo_continent['ratio'] = geo_continent['revenue']/geo_rev
temp_list = []
for a,b in geo_continent.iterrows():
    if b['ratio'] <= 0.01:
        temp_list.append('others')
    else:
        temp_list.append(b['continent'])
geo_continent['continent'] = temp_list
    
geo_subcontinent = df_combined.groupby('geoNetwork.subContinent', as_index = False)['totals.transactionRevenue'].sum()
geo_subcontinent.columns = ['subcontinent' , 'revenue']

geo_rev = geo_subcontinent['revenue'].sum()
geo_subcontinent['ratio'] = geo_subcontinent['revenue']/geo_rev
temp_list = []
for a,b in geo_subcontinent.iterrows():
    if b['ratio'] <= 0.01:
        temp_list.append('others')
    else:
        temp_list.append(b['subcontinent'])
geo_subcontinent['subcontinent'] = temp_list

geo_country = df_combined.groupby('geoNetwork.country', as_index = False)['totals.transactionRevenue'].sum()
geo_country.columns = ['country' , 'revenue']

geo_rev = geo_country['revenue'].sum()
geo_country['ratio'] = geo_country['revenue']/geo_rev
temp_list = []
for a,b in geo_country.iterrows():
    if b['ratio'] <= 0.01:
        temp_list.append('others')
    else:
        temp_list.append(b['country'])
geo_country['country'] = temp_list

geo_nd = df_combined.groupby('geoNetwork.networkDomain', as_index = False)['totals.transactionRevenue'].sum()
geo_nd.columns = ['netdomain' , 'revenue']

geo_rev = geo_nd['revenue'].sum()
geo_nd['ratio'] = geo_nd['revenue']/geo_rev
temp_list = []
for a,b in geo_nd.iterrows():
    if b['ratio'] <= 0.01:
        temp_list.append('others')
    else:
        temp_list.append(b['netdomain'])
geo_nd['netdomain'] = temp_list
        
geo_city = df_combined.groupby('geoNetwork.city', as_index = False)['totals.transactionRevenue'].sum()
geo_city.columns = ['city' , 'revenue']

geo_rev = geo_city['revenue'].sum()
geo_city['ratio'] = geo_city['revenue']/geo_rev
temp_list = []
for a,b in geo_city.iterrows():
    if b['ratio'] <= 0.01:
        temp_list.append('others')
    else:
        temp_list.append(b['city'])
geo_city['city'] = temp_list

fig , axes = plt.subplots(2,3 , figsize = (15,5))
axes[0,0].bar(geo_continent['continent'] , geo_continent['revenue'] , color = 'coral')
axes[0,0].set_xlabel('continent')
axes[0,0].set_ylabel('Revenue')

axes[0,1].bar(geo_subcontinent['subcontinent'] , geo_subcontinent['revenue'] , color = 'coral')
axes[0,1].set_xlabel('subcontinent')
axes[0,1].set_ylabel('Revenue')

axes[0,2].bar(geo_country['country'] , geo_country['revenue'] , color = 'coral')
axes[0,2].set_xlabel('country')
axes[0,2].set_ylabel('Revenue')

axes[1,0].bar(geo_nd['netdomain'] , geo_nd['revenue'] , color = 'coral')
axes[1,0].set_xlabel('Network Domain')
axes[1,0].set_ylabel('Revenue')

axes[1,1].bar(geo_city['city'] , geo_city['revenue'] , color = 'coral')
axes[1,1].set_xlabel('City')
axes[1,1].set_ylabel('Revenue')
df_combined['PurchaseFlag'] = '0'
def chk_revenue(x):
    if x != 0:
        return '1'
    else:
        return '0'
df_combined['PurchaseFlag'] = df_combined['totals.transactionRevenue'].apply(chk_revenue)
#Analyzing patterns for revenue vs non-revenue purchases
# Box Plots
f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(16, 4))
ax1.set_title('Revenue - PageViews', fontsize=14)
sns.boxplot(x="PurchaseFlag", y="totals.pageviews", data=df_combined,  ax=ax1)
ax1.set_xlabel("Purchase/No purchase",size = 12,alpha=0.8)
ax1.set_ylabel("PageViews",size = 12,alpha=0.8)

ax2.set_title('Revenue - Bounces', fontsize=14)
sns.boxplot(x="PurchaseFlag", y="totals.bounces", data=df_combined,  ax=ax2)
ax2.set_xlabel("Purchase/No purchase",size = 12,alpha=0.8)
ax2.set_ylabel("Bounces",size = 12,alpha=0.8)

ax3.set_title('Revenue - Visits', fontsize=14)
sns.boxplot(x="PurchaseFlag", y="totals.visits", data=df_combined,  ax=ax3)
ax3.set_xlabel("Purchase/No purchase",size = 12,alpha=0.8)
ax3.set_ylabel("Visits",size = 12,alpha=0.8)

ax4.set_title('Revenue - Hits', fontsize=14)
sns.boxplot(x="PurchaseFlag", y="totals.hits", data=df_combined,  ax=ax4)
ax4.set_xlabel("Purchase/No purchase",size = 12,alpha=0.8)
ax4.set_ylabel("Hits",size = 12,alpha=0.8)
#Starting with numerical columns alone
exclude_list = ['visitId' , 'visitStartTime' , 'visitNumber']
num_cols = [x for x in df_combined.columns if df_combined[x].dtype != 'object' and x not in exclude_list]
#sns.scatterplot('totals.pageviews' , 'totals.transactionRevenue' , data = df_combined , hue = 'PurchaseFlag')
df_combined['pageviews.log1x'] = np.log1p(df_combined['totals.pageviews'])
df_combined['visits.log1x'] = np.log1p(df_combined['totals.visits'])
df_combined['newvisits.log1x'] = np.log1p(df_combined['totals.visits'])
df_combined['hits.log1x'] = np.log1p(df_combined['totals.hits'])
df_combined['log_rev'] = np.log1p(df_combined['totals.transactionRevenue'])
#df_combined['pageviews.log1x'].head(10)
f , axes = plt.subplots(2,2 , figsize = (20,16))
axes[0,0].scatter('pageviews.log1x' , 'log_rev', data = df_combined)
axes[0,1].scatter('visits.log1x' , 'log_rev', data = df_combined)
axes[1,0].scatter('newvisits.log1x' , 'log_rev', data = df_combined)
axes[1,1].scatter('hits.log1x' , 'log_rev', data = df_combined)
#Small preprocessing steps
constant_columns = constant_cols(df_combined)
df_combined = df_combined.drop(constant_columns , axis = 1)
df_combined = to_lower(df_combined)
df_combined = replace_missing_vals(df_combined)
num_of_cat_levels=chk_num_lvls(df_combined)
df_combined = find_req_levels(df_combined)
df_combined = cat2num(df_combined)

#Split back into train and test datasets
train = df_combined[:train_length]
test = df_combined[train_length:]

X_train = train[train['date'].dt.date <= datetime.date(2017,5,31)]
X_val = train[train['date'].dt.date > datetime.date(2017,5,31)]
y_train = X_train['totals.transactionRevenue']
y_val = X_val['totals.transactionRevenue']
y_test = test['totals.transactionRevenue']
del X_train['totals.transactionRevenue']
del X_val['totals.transactionRevenue']
del test['totals.transactionRevenue']

cols_to_drop = ['date' , 'fullVisitorId' , 'sessionId' , 'visitId']
X_train = X_train.drop(cols_to_drop , axis = 1)
X_val = X_val.drop(cols_to_drop , axis = 1)
test = test.drop(cols_to_drop, axis = 1)

first_model_cols = ['totals.hits','totals.pageviews']
X_train = X_train[first_model_cols]
X_val = X_val[first_model_cols]
test = test[first_model_cols]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train , columns = ['hits','pageviews'])
X_val = scaler.transform(X_val)
X_val = pd.DataFrame(X_val , columns = ['hits','pageviews'])
test = scaler.transform(test)
test = pd.DataFrame(test,columns = ['hits','pageviews'])

import keras

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
#Model 1
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential

model = Sequential()
model.add(Dense(200 , input_dim = X_train.shape[1], activation='relu'))
model.add(Dropout(.2))
model.add(Activation('linear'))
model.add(Dense(150 , activation = 'relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])


y_train = np.log1p(y_train)
y_val = np.log1p(y_val)
y_test = np.log1p(y_test)

np.random.seed(5)
model.fit(X_train , y_train , nb_epoch = 150 ,batch_size = 50000, validation_data = [X_val , y_val], callbacks = [plot_losses] , verbose = 0)
preds = model.predict(test)

test_data = df_combined[train_length:]
test_data = test_data[['fullVisitorId']]
test_data['PredLogRev'] = preds
test_data = test_data.groupby("fullVisitorId")["PredLogRev"].sum().reset_index()
test_data.columns = ["fullVisitorId", "PredictedLogRevenue"]
#READING SUMISSION FILE
submission=pd.read_csv('../input/sample_submission.csv')

#CREATING JOIN BETWEEN PREDICTED DATA WITH SUBMISSION FILE
submission=submission.join(test_data.set_index('fullVisitorId'),on='fullVisitorId',lsuffix='_sub')
submission.drop('PredictedLogRevenue_sub',axis=1,inplace=True)

#HANDLING NaN IN CASE OF MISSING fullVisitorId
submission.fillna(0,inplace=True)

#SUBMITING FILE
submission.to_csv('storeRev_1_submission.csv',index=False)


