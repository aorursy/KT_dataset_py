import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
current_palette = sns.color_palette()
# from tqdm import tqdm_notebook
%matplotlib inline 
# import io
# from google.colab import files
!pip install fbprophet
from fbprophet import Prophet
# !pip install -U -q PyDrive
 
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
 
# 1. Authenticate and create the PyDrive client.
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# file_list = drive.ListFile({'q': "'1W1X7lXaLjCLODvQqZUjolbT8MiOhZCnN' in parents and trashed=false"}).GetList()
# for file1 in file_list:
#   print('title: %s, id: %s' % (file1['title'], file1['id']))
# transactions = drive.CreateFile({'id': '1VfZXGEi9Jx3iHhzYVxutNEKH0PMgfh6l'})
# transactions.GetContentFile('sales_train.csv.gz')

# items = drive.CreateFile({'id': '1YZrfzt-iDNGAJ7fAZZNMqGuS4OyJu1Nt'})
# items.GetContentFile('items.csv')

# item_categories = drive.CreateFile({'id': '1HJa-lyGoa7pfW-BXS8ckPQRMaYPvKbwq'})
# item_categories.GetContentFile('item_categories.csv')

# shops = drive.CreateFile({'id': '1jXGSJANRuhjhQ2AngVMQx9mRf911Cj3E'})
# shops.GetContentFile('shops.csv')

# transactions_test = drive.CreateFile({'id': '1WOkxqBGgUnZtJ5Zw83wADWG5cGkX0caX'})
# transactions_test.GetContentFile('test.csv.gz')
transactions    = pd.read_csv('../input/sales_train.csv')
items           = pd.read_csv('../input/items.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
shops           = pd.read_csv('../input/shops.csv')
transactions_test = pd.read_csv('../input/test.csv')
transactions.columns
transactions_test.columns
transactions.head()
items.head()
shops.head()
item_categories.head()
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=transactions.item_cnt_day)

plt.figure(figsize=(10,4))
plt.xlim(transactions.item_price.min(), transactions.item_price.max()*1.1)
sns.boxplot(x=transactions.item_price)
transactions = transactions[transactions.item_price<100000]
transactions = transactions[transactions.item_cnt_day<1001]
median = transactions[(transactions.shop_id==32)&(transactions.item_id==2973)&(transactions.date_block_num==4)&(transactions.item_price>0)].item_price.median()
transactions.loc[transactions.item_price<0, 'item_price'] = median
# Якутск Орджоникидзе, 56
transactions.loc[transactions.shop_id == 0, 'shop_id'] = 57
transactions_test.loc[transactions_test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
transactions.loc[transactions.shop_id == 1, 'shop_id'] = 58
transactions_test.loc[transactions_test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
transactions.loc[transactions.shop_id == 10, 'shop_id'] = 11
transactions_test.loc[transactions_test.shop_id == 10, 'shop_id'] = 11
'''treating datetime and adding revenue'''
transactions["new_date"] = pd.to_datetime(transactions.date, format='%d.%m.%Y')
transactions["month"] = pd.DatetimeIndex(transactions.new_date).month
time_range = pd.date_range(start='2013-01-01',
                           end='2015-10-30')
time_range = pd.DataFrame({'new_date': time_range})
time_range = time_range.set_index('new_date')
time_range.head(2)
transactions_test.shape
transactions_transf = transactions[['item_id', 'shop_id', 'item_cnt_day', 'new_date']]
transactions_transf = transactions_test.join(transactions_transf.set_index(['item_id', 'shop_id']), on=['item_id', 'shop_id'])
trans_lst = [x for _, x in transactions_transf.groupby(['item_id', 'shop_id'], as_index=False)]
len(trans_lst)
trans_lst[2:4]
len(trans_lst)
time_range = pd.date_range(start='2013-01-01',
                           end='2015-10-30')
time_range = pd.DataFrame({'new_date': time_range})
time_range = time_range.set_index('new_date')
def compute_res_df(current_df,
                   time_range=pd.DataFrame({'new_date': pd.date_range(start='2013-01-01',
                           end='2015-10-31')}).set_index('new_date')):
    shop_id = current_df.shop_id.iloc[0]
    item_id = current_df.item_id.iloc[0]
    
    ph_df = current_df[['new_date', 'item_cnt_day']].copy()
    
    df_train = ph_df[ph_df.new_date < '2015-10-01'].copy()
    df_test = ph_df[ph_df.new_date >= '2015-10-01'].copy()
    
    ph_df = df_train.copy()

    ph_df = time_range.join(ph_df.set_index("new_date")).fillna(0)

    ph_df = ph_df.reset_index().rename(columns={'new_date': 'ds',
                        'item_cnt_day': 'y'})

    m = Prophet(daily_seasonality=True)
    m.fit(ph_df)
    future = m.make_future_dataframe(periods=31)
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat']].rename(columns={'ds':'new_date',
                        'yhat':'target'})
    #     print(forecast.tail())  
    target_sum = np.max((np.round(forecast.target.sum()), 0))
    #     print(target_sum)
    res_df = pd.DataFrame({'shop_id': shop_id,
                           'item_id': item_id,
                           'target': target_sum},
                           index = [0])


    print("result df : ")
    print(res_df)
    
    print("df_test: ")
    print(df_test)
    
    
    return res_df
res_lst = []
i = 0
for df in trans_lst[:20]:
  i += 1
  print("Current_iter = ", i)
  res_df = compute_res_df(df)
  res_lst.append(res_df)
res_df = pd.concat(res_lst)
res_df.head()
res_df.to_csv("res_df.csv")

# Create & upload a file.
uploaded = drive.CreateFile({'res_df': 'res_df.csv'})
uploaded.SetContentFile('res_df.csv')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))